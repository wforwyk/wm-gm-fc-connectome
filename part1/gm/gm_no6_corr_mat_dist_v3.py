#!/usr/bin/env python3
"""
gm_no6_corr_mat_dist_v2.py
==========================
Per subject:
  1. Final GM voxel set from the binarised mask and temporal variance
  2. Inter-voxel FC correlation matrix        -> {subj}_R.npz      ('arr_0')
  3. Geodesic (Dijkstra) distance matrix      -> {subj}_W_a.npz    ('arr_0')
  4. Voxel set and coordinates                -> {subj}_vset.npz
  5. Updated GM mask                          -> bnbmdenoised_detrended_rest_{subj}.nii
  6. QC row                                   -> gm_no6_qc.csv

Original logic by Joshua Goh et al.  This version keeps the outputs and
their meaning identical while fixing four problems in the v1 port.

FIXES
  1. DIJKSTRA SCOPE.  v1 ran Dijkstra from every node of the full image
     lattice (64*64*32 = 131,072 sources) and allocated a 131,072^2
     float16 memmap, which is 34 GB on disk.  Only the grey-matter voxels
     are ever used.  This version builds a compact graph over the GM voxel
     set alone, so the work drops by roughly the ratio of lattice size to
     GM size, and the distance matrix is written directly at its final
     size.

  2. ADJACENCY CONSTRUCTION.  v1 located each voxel's coordinates with
     np.where(I == v), a full scan of the 131,072-element index array
     three times per voxel.  Replaced by np.unravel_index and a vectorised
     neighbour table.

  3. CORNER-STEP DISTANCE.  v1 gave the (+-1,+-1,+-1) diagonal step a
     length of sqrt(4^2 + 3.4^2 + 4^2) = 6.600 mm.  For a 3.4 x 3.4 x 4 mm
     voxel the correct length is sqrt(3.4^2 + 3.4^2 + 4^2) = 6.255 mm, a
     5.5% overestimate on every corner step.  Step lengths are now derived
     from the NIfTI header, so they are correct for any voxel size.

  4. ISOLATED CLUSTERS.  Voxels in small disconnected components have
     infinite geodesic distance to most of the brain.  v1 left them in
     R.npz and W_a.npz, and no7 filtered them only when building its own
     vectors, so the Part 2 scripts (which load R.npz and W_a.npz
     directly) still saw them.  They are now removed here, once, so that
     R, W_a, vset and the saved mask are mutually consistent.

  Predecessor paths (node_*.npz) are no longer written by default: v1
  produced one file per lattice node, i.e. 131,072 files per subject, and
  they are used only for path visualisation.

INPUT   {BASE_DIR}/{subj}/gm/func/derived/denoised_detrended_rest_{subj}.nii
        {BASE_DIR}/{subj}/gm/func/derived/bmdenoised_detrended_rest_{subj}.nii
NEXT    gm_no7_indices_and_output.py
"""

# ============================================================
# USER SETTINGS
# ============================================================
BASE_DIR = "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch"
SUBJECTS = ["1123", "1170"]        # subject id must be in " " and divided by ,

# Non-empty path overrides SUBJECTS: one subject ID per line, no quotes.
SUBJECTS_TXT = ""

# Skip subjects whose R.npz, W_a.npz and vset.npz all already exist.  Lets an
# interrupted batch be resumed without recomputing what finished.
SKIP_EXISTING = True

# Storage precision for the two large matrices.
#   float32 : 12,388^2 x 4 B = 614 MB each before compression
#   float16 : half that, at ~0.1 mm resolution for distance and ~5e-4 for r
# At 336 subjects the difference is roughly 235 GB versus 120 GB, so check
# free disk before launching the full batch.
STORE_DTYPE = "float32"

# Components smaller than this are removed from the voxel set.  Their
# geodesic distance to the rest of the brain is infinite, so they cannot
# enter any distance-dependent analysis.  Set to 0 to keep everything.
MIN_COMPONENT_SIZE = 100

# Source voxels per Dijkstra block.  Lower this if memory is tight.
DIJKSTRA_BLOCK = 500

# Write predecessor arrays for path visualisation (one file per GM voxel).
SAVE_PREDECESSORS = False
# ============================================================

import os
import gc
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import sparse, ndimage
from scipy.sparse.csgraph import dijkstra, connected_components


# ── paths ────────────────────────────────────────────────────
def gm_derived(subj):
    return os.path.join(BASE_DIR, subj, "gm", "func", "derived")


def dist_derived(subj):
    return os.path.join(BASE_DIR, subj, "gm", "anat", "derived", "dist_results")


# ── neighbour table ──────────────────────────────────────────
def neighbour_offsets(zooms):
    """
    The 26 neighbour offsets and their Euclidean step lengths in mm,
    derived from the image voxel sizes.
    """
    offs, dists = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                offs.append((di, dj, dk))
                dists.append(float(np.sqrt((di * zooms[0]) ** 2 +
                                           (dj * zooms[1]) ** 2 +
                                           (dk * zooms[2]) ** 2)))
    return np.array(offs, dtype=np.int64), np.array(dists, dtype=np.float64)


def build_compact_graph(mask, zooms):
    """
    Sparse 26-connected adjacency over the masked voxels only.
    Returns (A, coords) where A is n x n and coords is n x 3 in voxel units,
    both ordered by the C-order flat index of the mask, which is the order
    mask-based indexing (data[mask]) produces.
    """
    coords = np.argwhere(mask)                       # C-order, matches data[mask]
    n = len(coords)
    pos = -np.ones(mask.shape, dtype=np.int64)
    pos[tuple(coords.T)] = np.arange(n)

    offs, dists = neighbour_offsets(zooms)
    rows, cols, vals = [], [], []
    shape = np.array(mask.shape)

    for (off, d) in zip(offs, dists):
        nb = coords + off
        ok = np.all((nb >= 0) & (nb < shape), axis=1)
        if not ok.any():
            continue
        src = np.flatnonzero(ok)
        tgt = pos[tuple(nb[ok].T)]
        keep = tgt >= 0
        if not keep.any():
            continue
        rows.append(src[keep])
        cols.append(tgt[keep])
        vals.append(np.full(int(keep.sum()), d))

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, coords


# ── per-subject pipeline ─────────────────────────────────────
def run_subject(subj):
    print(f"\n{'=' * 60}\n  {subj}\n{'=' * 60}", flush=True)
    os.makedirs(os.path.join(gm_derived(subj), "fc_results"), exist_ok=True)
    os.makedirs(dist_derived(subj), exist_ok=True)

    func_path = os.path.join(gm_derived(subj), f"denoised_detrended_rest_{subj}.nii")
    mask_path = os.path.join(gm_derived(subj), f"bmdenoised_detrended_rest_{subj}.nii")
    for p in (func_path, mask_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # ── 1. voxel set ─────────────────────────────────────────
    print("  [1/5] Building voxel set...", flush=True)
    fvol = nib.load(func_path)
    Y = fvol.get_fdata(dtype=np.float32)
    zooms = np.asarray(fvol.header.get_zooms()[:3], dtype=np.float64)

    mvol = nib.load(mask_path)
    M = np.asarray(mvol.dataobj) > 0

    with np.errstate(invalid="ignore"):
        S = np.var(Y, axis=3)
    finite = ~np.any(~np.isfinite(Y), axis=3)
    keep = M & finite & (S > 0)

    n0 = int(keep.sum())
    print(f"    mask {int(M.sum())} -> finite & var>0: {n0} voxels "
          f"({n0 * float(np.prod(zooms)) / 1000:.1f} cm^3, "
          f"voxel {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm)", flush=True)
    if n0 == 0:
        raise ValueError("Empty voxel set.")

    # ── 2. connectivity structure, drop isolated clusters ────
    lab, ncomp = ndimage.label(keep, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    largest_pct = 100.0 * sizes.max() / n0
    print(f"    components: {ncomp} | largest {largest_pct:.1f}% | "
          f"isolated voxels {int((sizes == 1).sum())}", flush=True)
    if largest_pct < 90:
        print("    WARNING: mask is fragmented. Check the gm_no5 binarisation.",
              file=sys.stderr, flush=True)

    if MIN_COMPONENT_SIZE > 0:
        small = np.flatnonzero(sizes < MIN_COMPONENT_SIZE) + 1
        if small.size:
            drop = np.isin(lab, small)
            keep = keep & ~drop
            print(f"    dropped {int(drop.sum())} voxels in {small.size} components "
                  f"smaller than {MIN_COMPONENT_SIZE}", flush=True)

    n = int(keep.sum())
    print(f"    final n_gm = {n}", flush=True)

    # ── 3. save the mask actually used ───────────────────────
    out_mask = os.path.join(gm_derived(subj),
                            f"bnbmdenoised_detrended_rest_{subj}.nii")
    nib.save(nib.Nifti1Image(keep.astype(np.uint8), mvol.affine, mvol.header),
             out_mask)
    print(f"    saved {os.path.basename(out_mask)}", flush=True)

    # ── 4. FC correlation matrix ─────────────────────────────
    print("  [2/5] Computing FC correlation matrix...", flush=True)
    MY = Y[keep]                                   # [n x nT], C-order rows
    del Y
    gc.collect()
    R = np.corrcoef(MY).astype(STORE_DTYPE)
    r_path = os.path.join(gm_derived(subj), "fc_results", f"{subj}_R.npz")
    np.savez_compressed(r_path, R)
    print(f"    R shape {R.shape} -> {os.path.basename(r_path)}", flush=True)
    del R, MY
    gc.collect()

    # ── 5. adjacency over the GM voxels only ─────────────────
    print("  [3/5] Building compact adjacency...", flush=True)
    A, coords = build_compact_graph(keep, zooms)
    print(f"    A shape {A.shape}, {A.nnz} edges "
          f"({A.nnz / max(n, 1):.1f} per voxel)", flush=True)
    sparse.save_npz(os.path.join(dist_derived(subj), f"{subj}_A.npz"), A.tocsc())

    ncc, cc_lab = connected_components(A, directed=False)
    print(f"    graph components: {ncc}", flush=True)

    # ── 6. Dijkstra, blocked, written straight to final size ─
    print(f"  [4/5] Dijkstra over {n} sources "
          f"(blocks of {DIJKSTRA_BLOCK})...", flush=True)
    W = np.empty((n, n), dtype=STORE_DTYPE)
    pred_dir = os.path.join(dist_derived(subj), "path")
    if SAVE_PREDECESSORS:
        os.makedirs(pred_dir, exist_ok=True)

    for start in range(0, n, DIJKSTRA_BLOCK):
        idx = np.arange(start, min(start + DIJKSTRA_BLOCK, n))
        if SAVE_PREDECESSORS:
            d, pr = dijkstra(A, directed=False, indices=idx,
                             return_predecessors=True)
            for row, i in enumerate(idx):
                np.savez_compressed(os.path.join(pred_dir, f"node_{i}.npz"),
                                    pr[row])
        else:
            d = dijkstra(A, directed=False, indices=idx)
        W[idx, :] = d.astype(STORE_DTYPE)
        print(f"    {min(start + DIJKSTRA_BLOCK, n)}/{n}", end="\r", flush=True)
    print(flush=True)

    # Row-wise accumulation: a boolean mask over the whole n x n matrix
    # would allocate another n^2 array on top of W.
    n_inf = 0
    max_finite = 0.0
    for start in range(0, n, DIJKSTRA_BLOCK):
        blk = W[start:start + DIJKSTRA_BLOCK]
        fin = np.isfinite(blk)
        n_inf += int((~fin).sum())
        if fin.any():
            max_finite = max(max_finite, float(blk[fin].max()))
    frac_inf = n_inf / float(n) ** 2
    print(f"    unreachable pairs: {100 * frac_inf:.3f}% | "
          f"max finite distance {max_finite:.1f} mm", flush=True)

    w_path = os.path.join(dist_derived(subj), f"{subj}_W_a.npz")
    np.savez_compressed(w_path, W)
    print(f"    saved {os.path.basename(w_path)}", flush=True)

    # ── 7. voxel set record ──────────────────────────────────
    flat = np.flatnonzero(keep.ravel())
    np.savez_compressed(os.path.join(gm_derived(subj), f"{subj}_vset.npz"),
                        flat_index=flat, coords=coords.astype(np.int16),
                        shape=np.array(keep.shape), zooms=zooms,
                        affine=mvol.affine)

    del W
    gc.collect()
    print(f"  [5/5] {subj} done.", flush=True)

    return {"subject": subj, "n_gm_before_cleanup": n0, "n_gm": n,
            "volume_cm3": round(n * float(np.prod(zooms)) / 1000, 1),
            "n_components_mask": int(ncomp),
            "largest_component_pct": round(largest_pct, 2),
            "n_components_graph": int(ncc),
            "frac_unreachable_pairs": round(frac_inf, 6),
            "max_geodesic_mm": round(float(finite_W.max()), 2),
            "voxel_mm": "x".join(f"{z:.2f}" for z in zooms)}


# ── main ─────────────────────────────────────────────────────
def already_done(subj):
    return all(os.path.exists(p) for p in (
        os.path.join(gm_derived(subj), "fc_results", f"{subj}_R.npz"),
        os.path.join(dist_derived(subj), f"{subj}_W_a.npz"),
        os.path.join(gm_derived(subj), f"{subj}_vset.npz")))


def main():
    subjects = SUBJECTS
    if SUBJECTS_TXT:
        if not os.path.exists(SUBJECTS_TXT):
            raise FileNotFoundError(SUBJECTS_TXT)
        with open(SUBJECTS_TXT) as f:
            subjects = [ln.strip() for ln in f if ln.strip()]
        print(f"[info] {len(subjects)} subjects from {SUBJECTS_TXT}")

    if SKIP_EXISTING:
        pending = [s for s in subjects if not already_done(s)]
        n_skip = len(subjects) - len(pending)
        if n_skip:
            print(f"[info] skipping {n_skip} subjects that already have "
                  f"R.npz, W_a.npz and vset.npz")
        subjects = pending

    rows, failed = [], []
    for i, subj in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}]", flush=True)
        try:
            rows.append(run_subject(subj))
        except Exception as e:
            print(f"\n[ERROR] {subj}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            failed.append(subj)

    if rows:
        qc = pd.DataFrame(rows)
        qc_path = os.path.join(BASE_DIR, "gm_no6_qc.csv")
        # Append rather than overwrite, so a resumed batch keeps earlier rows.
        if os.path.exists(qc_path):
            old = pd.read_csv(qc_path)
            old = old[~old.subject.astype(str).isin(qc.subject.astype(str))]
            qc = pd.concat([old, qc], ignore_index=True)
        qc = qc.sort_values("subject")
        qc.to_csv(qc_path, index=False)
        print("\n" + qc.to_string(index=False))
        print(f"\nQC written: {qc_path}")

    if failed:
        print(f"\n[fail] {len(failed)} subjects: {', '.join(failed)}",
              file=sys.stderr)
    else:
        print("\n[fail] none")


if __name__ == "__main__":
    main()
