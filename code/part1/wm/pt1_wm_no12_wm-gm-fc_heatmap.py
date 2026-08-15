#!/usr/bin/env python3
"""
no12_wm-gm-fc_heatmap.py
==========================
PURPOSE : Generate GM voxel x voxel heatmaps for FC, geodesic distance,
          and WM radial diffusivity (RD). Loops over multiple subjects.

PIPELINE CONTEXT:
  no11 → {subj}_voxelpair_RD.csv       (idx_u, idx_v, RD per voxel pair)
  R.npz, W_a.npz                       (FC and geodesic dist matrices)
      ↓ shared sampling (seed=42)
  → heatmap_A_fc_{subj}.png            (FC matrix,   RdBu_r)
  → heatmap_B_dist_{subj}.png          (Geodesic dist, Greys)
  → heatmap_C_rd_{subj}.png            (RD voxel×voxel, Blues)

INPUT   : fc           — {subj}_R.npz
          dist         — {subj}_W_a.npz
          voxelpair_rd — wm-gm-fc/3d/{subj}_voxelpair_RD.csv  (from no11)

OUTPUT  : {BASE_DIR}/{subj}/wm-gm-fc/heatmap/heatmap_{A,B,C}_{subj}.png

NOTE    : Numeric subject IDs only (no sub- prefix) for batch.
          voxelpair_rd CSV requires columns: idx_u, idx_v, RD.
          SAMPLE_N controls how many voxels are sampled per axis.
          Same random seed (42) used across all three panels.

          Panel C is sparse by construction: the SAMPLE_N voxels drawn here
          and the per-region voxels no11 sampled (MAX_VOXELS) are independent
          draws from the same voxel set, so only their intersection can be
          filled.  The fill percentage is printed; if it is near zero, raise
          SAMPLE_N and/or no11's MAX_VOXELS.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "0001", "0002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/path/to/your/project/derivatives/batch")

SAMPLE_N = 500   # voxels per axis
DPI      = 300
# ────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_sample_idx(total, n, seed=42):
    if total <= n:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, n, replace=False))


def load_fc(path, idx):
    print(f"  Loading FC ...", end=" ", flush=True)
    arr = np.load(path)["arr_0"].astype(np.float32)
    print(f"shape={arr.shape}", end=" -> ", flush=True)
    sub = arr[np.ix_(idx, idx)]
    print(f"sampled {sub.shape}")
    return sub


def load_dist(path, idx):
    print(f"  Loading dist ...", end=" ", flush=True)
    arr = np.load(path)["arr_0"].astype(np.float32)
    print(f"shape={arr.shape}", end=" -> ", flush=True)
    sub = arr[np.ix_(idx, idx)]
    sub[sub <= 0] = np.nan
    print(f"sampled {sub.shape}")
    return sub


def build_rd_matrix(csv_path, idx):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = {"idx_u", "idx_v", "RD"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    print(f"  Loaded voxelpair CSV: {len(df):,} pairs")

    idx_set = set(idx.tolist())
    idx_pos = {v: i for i, v in enumerate(idx)}
    n = len(idx)

    mask = df["idx_u"].isin(idx_set) & df["idx_v"].isin(idx_set)
    sub  = df[mask]
    rd_mat = np.full((n, n), np.nan, dtype=np.float32)
    grp = sub.groupby(["idx_u", "idx_v"])["RD"].mean()
    for (iu, iv), val in grp.items():
        ri = idx_pos.get(int(iu))
        ci = idx_pos.get(int(iv))
        if ri is not None and ci is not None:
            rd_mat[ri, ci] = val
            rd_mat[ci, ri] = val
    pct = 100 * np.sum(np.isfinite(rd_mat)) / rd_mat.size
    print(f"  RD matrix {n}x{n}: {pct:.1f}% filled")
    return rd_mat


def save_heatmap(mat, title, xlabel, ylabel, clabel, cmap, fname,
                 center=False):
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        print(f"  SKIP {Path(fname).name}: no finite values to plot")
        return

    fig, ax = plt.subplots(figsize=(7, 6.5))
    vmin = np.percentile(finite, 1)
    vmax = np.percentile(finite, 99)
    if center:
        vmax = np.percentile(np.abs(finite), 99)
        vmin = -vmax
    im = ax.imshow(mat, aspect="equal", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(clabel, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {Path(fname).name}")


def run_subject(subj):
    fc           = f"{BASE_DIR}/{subj}/gm/func/derived/fc_results/{subj}_R.npz"
    dist         = f"{BASE_DIR}/{subj}/gm/anat/derived/dist_results/{subj}_W_a.npz"
    voxelpair_rd = f"{BASE_DIR}/{subj}/wm-gm-fc/3d/{subj}_voxelpair_RD.csv"
    out          = Path(BASE_DIR) / subj / "wm-gm-fc"/ "heatmap"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n[{subj}]")
    print(f"  Probing R.npz shape ...", end=" ", flush=True)
    n_total = np.load(fc)["arr_0"].shape[0]
    idx = get_sample_idx(n_total, SAMPLE_N)
    print(f"n_total={n_total}  sample={len(idx)}")

    # A: FC
    print("  [A] FC matrix")
    R_sub = load_fc(fc, idx)
    save_heatmap(
        R_sub,
        title=f"A.  Functional Connectivity  (R.npz)\nsub-{subj}  --  {R_sub.shape[0]}x{R_sub.shape[1]} sample",
        xlabel="GM voxel (sampled)", ylabel="GM voxel (sampled)",
        clabel="Pearson r", cmap="RdBu_r", center=True,
        fname=str(out / f"heatmap_A_fc_{subj}.png"),
    )
    del R_sub

    # B: Geodesic distance
    print("  [B] Geodesic distance matrix")
    W_sub = load_dist(dist, idx)
    save_heatmap(
        W_sub,
        title=f"B.  Geodesic Distance  (W_a.npz)\nsub-{subj}  --  {W_sub.shape[0]}x{W_sub.shape[1]} sample",
        xlabel="GM voxel (sampled)", ylabel="GM voxel (sampled)",
        clabel="Geodesic distance (mm)", cmap="Greys",
        fname=str(out / f"heatmap_B_dist_{subj}.png"),
    )
    del W_sub

    # C: RD voxel x voxel
    print("  [C] RD voxel x voxel matrix")
    rd_mat = build_rd_matrix(voxelpair_rd, idx)
    save_heatmap(
        rd_mat,
        title=f"C.  WM Radial Diffusivity  (RD)\nsub-{subj}  --  {len(idx)}x{len(idx)} sample",
        xlabel="GM voxel (sampled)", ylabel="GM voxel (sampled)",
        clabel="RD (mm2/s)", cmap="Blues",
        fname=str(out / f"heatmap_C_rd_{subj}.png"),
    )
    print(f"  Done -> {out.resolve()}")


def main():
    print("=" * 52)
    print("  no12_wm-gm-fc_heatmap.py")
    print(f"  Subjects : {len(SUBJECTS)}")
    print("=" * 52)

    failed = []
    for subj in SUBJECTS:
        try:
            run_subject(subj)
        except Exception as e:
            print(f"  ERROR [{subj}]: {e}")
            failed.append(subj)

    print("\n" + "=" * 52)
    if not failed:
        print("  All done. No failures.")
    else:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    {f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
