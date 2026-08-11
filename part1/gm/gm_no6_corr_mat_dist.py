#!/usr/bin/env python3
"""
no6_corr_mat_dist.py
=====================
Computes per-subject:
  1. Inter-voxel FC correlation matrix  → {subj}_R.npz
  2. Updated GM mask (variance > 0)     → bnbmdenoised_detrended_rest_{subj}.nii
  3. Adjacency matrix (Dijkstra-ready)  → {subj}_A.npz
  4. Full Dijkstra distance matrix      → {subj}_W.dat  (memmap, temp)
  5. GM-masked distance matrix          → {subj}_W_a.npz
  6. Cleanup of temp W.dat

Original logic by Joshua Goh et al. — restructured for CLI batch use.
Output files and computation are unchanged.
"""

# ============================================================
# USER SETTINGS — edit only this section
# ============================================================
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

# Number of parallel processes for adjacency matrix computation
NUM_PROCESSES = 12
# ============================================================

import os
import gc
import numpy as np
import nibabel as nib
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from multiprocessing import Pool


# ── path helpers ────────────────────────────────────────────
def gm_derived(subj):
    return os.path.join(BASE_DIR, subj, "gm", "func", "derived")

def dist_derived(subj):
    return os.path.join(BASE_DIR, subj, "gm", "anat", "derived", "dist_results")

def dist_path(subj):
    return os.path.join(dist_derived(subj), "path")

def func_nii(subj):
    return os.path.join(gm_derived(subj), f"denoised_detrended_rest_{subj}.nii")

def mask_nii(subj):
    return os.path.join(gm_derived(subj), f"bmdenoised_detrended_rest_{subj}.nii")

def new_mask_nii(subj):
    return os.path.join(gm_derived(subj), f"bnbmdenoised_detrended_rest_{subj}.nii")

def r_npz(subj):
    return os.path.join(gm_derived(subj), "fc_results", f"{subj}_R.npz")

def a_npz(subj):
    return os.path.join(dist_derived(subj), f"{subj}_A.npz")

def w_dat(subj):
    return os.path.join(dist_derived(subj), f"{subj}_W.dat")

def w_a_npz(subj):
    return os.path.join(dist_derived(subj), f"{subj}_W_a.npz")


# ── adjacency function (unchanged from original) ────────────
def adj_cb(data, c, t):
    n = np.reshape([], (-1, 4))
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if not ((i == 0) and (j == 0) and (k == 0)) and \
                   (c[0]+i >= 0 and c[1]+j >= 0 and c[2]+k >= 0 and
                    c[0]+i < data.shape[0] and c[1]+j < data.shape[1] and
                    c[2]+k < data.shape[2] and
                    data[c[0]+i, c[1]+j, c[2]+k] > t):
                    if (i != 0) and (j != 0) and (k != 0):
                        r = (4**2 + 3.4**2 + 4**2) ** 0.5
                    elif len(np.where(np.array([i, j, k]) == 0)[0]) == 1:
                        if k == 0:
                            r = 3.4 * (2 ** 0.5)
                        else:
                            r = (4**2 + 3.4**2) ** 0.5
                    elif (i == 0) and (j == 0) and ((k == 1) or (k == -1)):
                        r = 4
                    else:
                        r = 3.4
                    n = np.concatenate(
                        (n, np.reshape(np.array([c[0]+i, c[1]+j, c[2]+k, r]), (-1, 4))),
                        axis=0,
                    )
    return n


# ── adjacency matrix worker (unchanged logic) ────────────────
def adj_mat(subject, MM, vset, I):
    print(f"  [adj_mat] {subject}", flush=True)
    num_voxels = np.prod(MM.shape)
    A = sparse.lil_matrix((num_voxels, num_voxels), dtype=np.float32)

    for v in vset:
        cc = np.array([
            np.where(I == v)[0][0],
            np.where(I == v)[1][0],
            np.where(I == v)[2][0],
        ])
        n = adj_cb(MM, cc, 0.5)
        for q in np.arange(0, n.shape[0]):
            adjacent_voxel = I[np.intc(n[q, 0]), np.intc(n[q, 1]), np.intc(n[q, 2])]
            A[v, adjacent_voxel] = n[q, 3]

    A = A.tocsc()
    sparse.save_npz(a_npz(subject), A)
    print(f"  [adj_mat] saved {a_npz(subject)}", flush=True)
    gc.collect()


# ── per-subject pipeline ─────────────────────────────────────
def run_subject(subj):
    print(f"\n{'='*60}\n  {subj}\n{'='*60}", flush=True)

    # make output dirs
    os.makedirs(os.path.join(gm_derived(subj), "fc_results"), exist_ok=True)
    os.makedirs(dist_derived(subj), exist_ok=True)
    os.makedirs(dist_path(subj), exist_ok=True)

    # ── 1. Load functional data ──────────────────────────────
    print(f"  [1/5] Loading functional data...", flush=True)
    brain_vol = nib.load(func_nii(subj))
    Y = brain_vol.get_fdata()
    S = np.var(Y, axis=3)
    Y = np.reshape(Y, (np.prod(Y.shape[:3]), Y.shape[3]))

    # ── 2. Build vset from GM mask ───────────────────────────
    print(f"  [2/5] Building GM vset...", flush=True)
    brain_mask_vol = nib.load(mask_nii(subj))
    M = brain_mask_vol.get_fdata()
    I = np.reshape(np.arange(np.prod(M.shape)), M.shape)
    vset = I[np.where((M != 0) & (S != 0))]
    MY = Y[vset, :]
    print(f"    vset size: {vset.shape[0]}", flush=True)

    # ── 3. Save updated GM mask (bnbm...) ────────────────────
    MM = np.zeros(M.shape)
    MM[np.where((M != 0) & (S != 0))] = 1
    MM_vol = nib.Nifti1Image(MM, brain_mask_vol.affine, brain_mask_vol.header)
    nib.save(MM_vol, new_mask_nii(subj))
    print(f"    saved {new_mask_nii(subj)}", flush=True)

    # ── 4. FC correlation matrix ─────────────────────────────
    print(f"  [3/5] Computing FC correlation matrix...", flush=True)
    R = np.corrcoef(MY)
    np.savez_compressed(r_npz(subj), R)
    print(f"    R shape: {R.shape}  saved {r_npz(subj)}", flush=True)
    del R, MY, Y
    gc.collect()

    # ── 5. Adjacency matrix ──────────────────────────────────
    print(f"  [4/5] Computing adjacency matrix (n_processes={NUM_PROCESSES})...", flush=True)
    if __name__ == "__main__":
        with Pool(processes=NUM_PROCESSES) as pool:
            pool.starmap(adj_mat, [(subj, MM, vset, I)])
    else:
        adj_mat(subj, MM, vset, I)
    del MM
    gc.collect()

    # ── 6. Dijkstra distance matrix ──────────────────────────
    print(f"  [5/5] Computing Dijkstra distances...", flush=True)
    A = sparse.load_npz(a_npz(subj))
    W = np.memmap(w_dat(subj), dtype="float16", mode="w+", shape=A.shape)

    for i in np.arange(0, A.shape[0]):
        W[i, :], predecessors = dijkstra(
            csgraph=A, directed=False, indices=i, return_predecessors=True
        )
        node_fp = os.path.join(dist_path(subj), f"node_{i}.npz")
        np.savez_compressed(node_fp, predecessors)

    print(f"    Dijkstra done. Extracting GM-masked W_a...", flush=True)

    # ── 7. Extract GM-masked distance matrix ─────────────────
    MW = W[vset][:, vset]
    np.savez(w_a_npz(subj), MW)
    print(f"    saved {w_a_npz(subj)}", flush=True)
    del W, MW, A
    gc.collect()

    # ── 8. Cleanup temp W.dat ────────────────────────────────
    if os.path.exists(w_dat(subj)):
        os.remove(w_dat(subj))
        print(f"    removed {w_dat(subj)}", flush=True)

    print(f"  [{subj}] Done.", flush=True)


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for subj in SUBJECTS:
        try:
            run_subject(subj)
        except Exception as e:
            print(f"\n[ERROR] {subj}: {e}", flush=True)

    print("\nAll subjects complete.")
