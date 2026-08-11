#!/usr/bin/env python3
"""
no7_indices_and_output.py
==========================
Derives flattened upper-triangle vectors from FC and geodesic distance
matrices, computes their correlation, and saves results.

Outputs per subject:
  vector_r_{subj}.npy   — upper-triangle FC values
  vector_w_{subj}.npy   — upper-triangle geodesic distance values
  (prints FC–distance correlation coefficient)

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

# Isolated voxel cluster size threshold (from original)
INF_ROW_THRESHOLD = 100
# ============================================================

import os
import gc
import numpy as np


# ── path helpers ────────────────────────────────────────────
def gm_dir(subj):
    return os.path.join(BASE_DIR, subj, "gm")

def path_r_npz(subj):
    return os.path.join(gm_dir(subj), "func", "derived", "fc_results", f"{subj}_R.npz")

def path_w_npz(subj):
    return os.path.join(gm_dir(subj), "anat", "derived", "dist_results", f"{subj}_W_a.npz")

def path_out_dir(subj):
    # save vectors alongside W_a.npz
    return os.path.join(gm_dir(subj), "anat", "derived", "dist_results")

def path_mem_r(subj):
    return os.path.join(gm_dir(subj), "func", "derived", "fc_results", f"{subj}_memR.dat")

def path_mem_w(subj):
    return os.path.join(gm_dir(subj), "anat", "derived", "dist_results", f"{subj}_memW.dat")


# ── per-subject logic (unchanged from original) ──────────────
def run_subject(subj):
    print(f"\n{'='*60}\n  {subj}\n{'='*60}", flush=True)

    # load R and W_a
    try:
        R = np.load(path_r_npz(subj))
        W = np.load(path_w_npz(subj))
    except FileNotFoundError as e:
        print(f"  [Error] {e}")
        return

    R_array = R["arr_0"]
    W_array = W["arr_0"]

    n = int(R_array.shape[0])

    # write to memmap for row-wise access
    open(path_mem_w(subj), "wb").close()
    open(path_mem_r(subj), "wb").close()

    matrix_w = np.memmap(path_mem_w(subj), dtype="float16", mode="r+", shape=(n, n))
    matrix_r = np.memmap(path_mem_r(subj), dtype="float16", mode="r+", shape=(n, n))
    matrix_r[:] = R_array[:]
    matrix_w[:] = W_array[:]
    del R_array, W_array, R, W
    gc.collect()

    n = matrix_w.shape[0]

    # remove isolated voxel rows (inf in W)
    inf_rows = [
        i for i in range(n)
        if np.isinf(matrix_w[i, :]).sum() > (n - INF_ROW_THRESHOLD)
    ]
    valid_rows = ~np.isin(np.arange(n), inf_rows)
    matrix_r_c = matrix_r[valid_rows][:, valid_rows]
    matrix_w_c = matrix_w[valid_rows][:, valid_rows]

    # extract upper triangle (exclude diagonal)
    upper_idx_r = np.triu_indices(matrix_r_c.shape[0], k=1)
    upper_idx_w = np.triu_indices(matrix_w_c.shape[0], k=1)
    vector_r = matrix_r_c[upper_idx_r].reshape(-1)
    vector_w = matrix_w_c[upper_idx_w].reshape(-1)

    # FC–distance correlation
    corr = np.corrcoef(vector_w, vector_r)[0, 1]
    print(f"  Correlation coefficient (FC–distance): {corr:.6f}", flush=True)

    # save vectors
    out_dir = path_out_dir(subj)
    np.save(os.path.join(out_dir, f"vector_r_{subj}.npy"), vector_r)
    np.save(os.path.join(out_dir, f"vector_w_{subj}.npy"), vector_w)
    print(f"  Saved vector_r and vector_w to {out_dir}", flush=True)

    del matrix_r, matrix_w, matrix_r_c, matrix_w_c, vector_r, vector_w
    gc.collect()

    # cleanup memmap temp files
    for p in [path_mem_r(subj), path_mem_w(subj)]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed {p}", flush=True)
        else:
            print(f"  Not found (skip): {p}", flush=True)

    print(f"  [{subj}] Done.", flush=True)


# ── main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for subj in SUBJECTS:
        try:
            run_subject(subj)
        except Exception as e:
            print(f"\n[ERROR] {subj}: {e}", flush=True)

    print("\nBatch processing complete.")
