#!/usr/bin/env python3
"""
gm_no7_fc_distance_summary_v2.py
================================
Subject-level summary of the FC-versus-geodesic-distance relationship.

WHAT CHANGED FROM v1 (no7_indices_and_output)

  1. ISOLATED-ROW FILTER REMOVED.  v1's central operation was dropping
     voxels whose geodesic distance was infinite to most of the brain:
         inf_rows = [i for i in range(n)
                     if np.isinf(matrix_w[i,:]).sum() > (n - 100)]
     That filter existed because the old binarisation fragmented the GM
     mask.  gm_no6_v2 removes small components once, at the source, and
     reports 0.000% unreachable pairs with a single connected graph, so
     there is nothing left to filter.  Keeping it also created an
     inconsistency: v1 filtered only its own vectors, while the Part 2
     scripts loaded R.npz and W_a.npz directly and still saw everything.

  2. NO MEMMAP ROUND TRIP.  v1 copied both matrices into float16 memmaps
     on disk, sliced them, then deleted the files.  The matrices are read
     directly here, in row blocks.

  3. NO GIANT VECTORS BY DEFAULT.  vector_r and vector_w hold every
     upper-triangle pair: at n = 12,388 that is 76.7 M values, about
     307 MB per subject, roughly 103 GB over 336 subjects, all of it a
     reshaping of data already stored in R.npz and W_a.npz.  The
     statistics are accumulated in row blocks instead, which is exact and
     uses constant memory.  Set SAVE_VECTORS to keep them anyway.

  4. MORE THAN ONE NUMBER.  v1 printed a single Pearson r and saved
     nothing else.  This version writes a row per subject containing the
     quantities the Part 2 analyses need, in both the raw-r form v1 used
     (so earlier results remain comparable) and the Fisher-Z form the
     project uses elsewhere, plus the log-distance fit that Aim 1 adopted.

OUTPUT
  {BASE_DIR}/gm_no7_fc_distance_summary.csv     one row per subject
  {BASE_DIR}/fc_distance_curves/{subj}_curve.csv   binned decay curve
  optional per-subject vector_r / vector_w

NOTE ON INTERPRETATION
  These are the subject-level GM functional-organisation measures.
  fc_distance_r reproduces the quantity that predicted CDRISC in the
  earlier regression; log_slope is the same relationship in the form
  Aim 1 fits.  They feed the individual-difference stage later, so no
  behavioural variable is touched here.
"""

# ============================================================
# USER SETTINGS
# ============================================================
BASE_DIR = "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch"
SUBJECTS = ["1123", "1170"]

# Non-empty path overrides SUBJECTS: one subject ID per line, no quotes.
SUBJECTS_TXT = ""

# Skip subjects already present in the summary CSV.
SKIP_EXISTING = True

# Rows of the matrices processed at a time.
ROW_BLOCK = 500

# Distance bins (mm) for the decay curve used in figures.
BIN_WIDTH_MM = 5.0

# Keep the full upper-triangle vectors.  Off by default: about 307 MB per
# subject at n = 12,388, and they duplicate R.npz and W_a.npz.
SAVE_VECTORS = False
# ============================================================

import os
import sys
import numpy as np
import pandas as pd


def gm_dir(subj):
    return os.path.join(BASE_DIR, subj, "gm")


def path_r(subj):
    return os.path.join(gm_dir(subj), "func", "derived", "fc_results", f"{subj}_R.npz")


def path_w(subj):
    return os.path.join(gm_dir(subj), "anat", "derived", "dist_results", f"{subj}_W_a.npz")


class PairAccumulator:
    """
    Exact sums over upper-triangle voxel pairs, accumulated block by block.

    Pearson correlation and least-squares slopes only need the counts,
    sums, sums of squares and cross products, so nothing has to be held in
    memory at full length.  All sums are float64 regardless of how the
    matrices are stored.
    """

    def __init__(self):
        self.n = 0
        self.s = {k: 0.0 for k in
                  ("w", "ww", "r", "rr", "wr", "z", "zz", "wz",
                   "lw", "lwlw", "lwz", "lwr")}
        self.n_nonfinite = 0

    def add(self, w, r):
        good = np.isfinite(w) & np.isfinite(r) & (w > 0)
        self.n_nonfinite += int((~good).sum())
        if not good.any():
            return
        w = w[good].astype(np.float64)
        r = r[good].astype(np.float64)
        z = np.arctanh(np.clip(r, -0.9999, 0.9999))
        lw = np.log(w)

        self.n += w.size
        self.s["w"] += w.sum();       self.s["ww"] += (w * w).sum()
        self.s["r"] += r.sum();       self.s["rr"] += (r * r).sum()
        self.s["wr"] += (w * r).sum()
        self.s["z"] += z.sum();       self.s["zz"] += (z * z).sum()
        self.s["wz"] += (w * z).sum()
        self.s["lw"] += lw.sum();     self.s["lwlw"] += (lw * lw).sum()
        self.s["lwz"] += (lw * z).sum()
        self.s["lwr"] += (lw * r).sum()

    def _corr(self, sx, sxx, sy, syy, sxy):
        n = self.n
        if n < 3:
            return np.nan
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        if vx <= 0 or vy <= 0:
            return np.nan
        return float(cov / np.sqrt(vx * vy))

    def _slope(self, sx, sxx, sy, sxy):
        n = self.n
        vx = sxx - sx * sx / n
        if n < 3 or vx <= 0:
            return np.nan, np.nan
        b = (sxy - sx * sy / n) / vx
        a = (sy - b * sx) / n
        return float(b), float(a)

    def result(self):
        s = self.s
        out = {"n_pairs": self.n, "n_pairs_dropped": self.n_nonfinite}
        # v1-comparable: raw r against linear distance
        out["fc_distance_r"] = self._corr(s["w"], s["ww"], s["r"], s["rr"], s["wr"])
        # Fisher-Z against linear distance
        out["fcz_distance_r"] = self._corr(s["w"], s["ww"], s["z"], s["zz"], s["wz"])
        # Fisher-Z against log distance, the form Aim 1 fits
        out["fcz_logdistance_r"] = self._corr(s["lw"], s["lwlw"], s["z"], s["zz"], s["lwz"])
        b, a = self._slope(s["lw"], s["lwlw"], s["z"], s["lwz"])
        out["log_slope"] = b
        out["log_intercept"] = a
        out["mean_distance_mm"] = s["w"] / self.n if self.n else np.nan
        out["mean_fc_r"] = s["r"] / self.n if self.n else np.nan
        out["mean_fc_z"] = s["z"] / self.n if self.n else np.nan
        return out


def run_subject(subj):
    for p in (path_r(subj), path_w(subj)):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    R = np.load(path_r(subj))["arr_0"]
    W = np.load(path_w(subj))["arr_0"]
    if R.shape != W.shape or R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R {R.shape} and W {W.shape} must be the same square matrix.")
    n = R.shape[0]
    print(f"  n_gm = {n}, upper-triangle pairs = {n * (n - 1) // 2:,}", flush=True)

    acc = PairAccumulator()
    nbin = int(np.ceil(300.0 / BIN_WIDTH_MM))
    bin_sum_z = np.zeros(nbin)
    bin_sum_r = np.zeros(nbin)
    bin_cnt = np.zeros(nbin, dtype=np.int64)

    vec_r, vec_w = ([], []) if SAVE_VECTORS else (None, None)

    for a in range(0, n, ROW_BLOCK):
        b = min(a + ROW_BLOCK, n)
        wblk = W[a:b].astype(np.float64)
        rblk = R[a:b].astype(np.float64)
        # keep only j > i so each pair is counted once
        rows = np.arange(a, b)[:, None]
        cols = np.arange(n)[None, :]
        upper = cols > rows
        w = wblk[upper]
        r = rblk[upper]
        acc.add(w, r)

        good = np.isfinite(w) & np.isfinite(r) & (w > 0)
        if good.any():
            wg, rg = w[good], r[good]
            zg = np.arctanh(np.clip(rg, -0.9999, 0.9999))
            idx = np.clip((wg / BIN_WIDTH_MM).astype(int), 0, nbin - 1)
            np.add.at(bin_sum_z, idx, zg)
            np.add.at(bin_sum_r, idx, rg)
            np.add.at(bin_cnt, idx, 1)
            if SAVE_VECTORS:
                vec_w.append(w.astype(np.float32))
                vec_r.append(r.astype(np.float32))
        print(f"    {b}/{n}", end="\r", flush=True)
    print(flush=True)

    res = acc.result()
    res["subject"] = subj
    res["n_gm"] = n

    # binned decay curve
    keep = bin_cnt > 0
    curve = pd.DataFrame({
        "subject": subj,
        "distance_mm": (np.arange(nbin)[keep] + 0.5) * BIN_WIDTH_MM,
        "n_pairs": bin_cnt[keep],
        "mean_fc_z": bin_sum_z[keep] / bin_cnt[keep],
        "mean_fc_r": bin_sum_r[keep] / bin_cnt[keep]})
    curve_dir = os.path.join(BASE_DIR, "fc_distance_curves")
    os.makedirs(curve_dir, exist_ok=True)
    curve.to_csv(os.path.join(curve_dir, f"{subj}_curve.csv"), index=False)

    if SAVE_VECTORS:
        out_dir = os.path.join(gm_dir(subj), "anat", "derived", "dist_results")
        np.save(os.path.join(out_dir, f"vector_r_{subj}.npy"), np.concatenate(vec_r))
        np.save(os.path.join(out_dir, f"vector_w_{subj}.npy"), np.concatenate(vec_w))

    print(f"  FC~distance   r = {res['fc_distance_r']:+.6f}   (v1-comparable)", flush=True)
    print(f"  FCz~logdist   r = {res['fcz_logdistance_r']:+.6f}   "
          f"slope = {res['log_slope']:+.6f}", flush=True)
    if res["n_pairs_dropped"]:
        print(f"  dropped {res['n_pairs_dropped']:,} non-finite or zero-distance pairs",
              flush=True)
    return res


def main():
    subjects = SUBJECTS
    if SUBJECTS_TXT:
        if not os.path.exists(SUBJECTS_TXT):
            raise FileNotFoundError(SUBJECTS_TXT)
        with open(SUBJECTS_TXT) as f:
            subjects = [ln.strip() for ln in f if ln.strip()]
        print(f"[info] {len(subjects)} subjects from {SUBJECTS_TXT}")

    summary_path = os.path.join(BASE_DIR, "gm_no7_fc_distance_summary.csv")
    done = set()
    if SKIP_EXISTING and os.path.exists(summary_path):
        done = set(pd.read_csv(summary_path).subject.astype(str))
        pending = [s for s in subjects if s not in done]
        if len(pending) < len(subjects):
            print(f"[info] skipping {len(subjects) - len(pending)} already summarised")
        subjects = pending

    rows, failed = [], []
    for i, subj in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subj}", flush=True)
        try:
            rows.append(run_subject(subj))
        except Exception as e:
            print(f"[ERROR] {subj}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            failed.append(subj)

    if rows:
        df = pd.DataFrame(rows)
        cols = ["subject", "n_gm", "n_pairs", "fc_distance_r", "fcz_distance_r",
                "fcz_logdistance_r", "log_slope", "log_intercept",
                "mean_distance_mm", "mean_fc_r", "mean_fc_z", "n_pairs_dropped"]
        df = df[[c for c in cols if c in df.columns]]
        if os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            old = old[~old.subject.astype(str).isin(df.subject.astype(str))]
            df = pd.concat([old, df], ignore_index=True)
        df = df.sort_values("subject")
        df.to_csv(summary_path, index=False)
        print(f"\nSummary written: {summary_path}")
        print(df.describe().T[["count", "mean", "std", "min", "max"]].to_string())

    if failed:
        print(f"\n[fail] {len(failed)} subjects: {', '.join(failed)}", file=sys.stderr)
    else:
        print("\n[fail] none")


if __name__ == "__main__":
    main()
