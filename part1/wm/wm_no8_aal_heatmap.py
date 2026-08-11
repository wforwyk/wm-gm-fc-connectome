#!/usr/bin/env python3
"""
no8_aal_heatmap.py
=========================
PURPOSE : Generate AAL-AAL weighted connectivity matrices per WM metric
          by combining pathstats (WM scalar) with AAL endpoint assignments.
          Produces edge detail CSVs and heatmap images.

PIPELINE CONTEXT:
  no6 → {subj}_pathstats.csv          (tract × WM metrics)
  no10 → {subj}_aal_summary.csv       (tract × endpoint × top3 AAL regions)
      ↓ weight = metric_avg × (top_i_percent/100 × top_j_percent/100)
  → aal_aal_{metric}_matrix_{subj}.csv  (AAL × AAL weighted matrix)
  → aal_edge_details_{metric}_{subj}.csv
  → aal_aal_{metric}_heatmap_{subj}.png

INPUT   : {BASE_DIR}/{subj}/wm/derived/pathstats/{subj}_pathstats.csv
          {BASE_DIR}/{subj}/wm/derived/aal_summary/{subj}_aal_summary.csv

OUTPUT  : {BASE_DIR}/{subj}/wm/derived/aal_summary/heatmap/
              aal_aal_{metric}_matrix_{subj}.csv
              aal_edge_details_{metric}_{subj}.csv
              aal_aal_{metric}_heatmap_{subj}.png

NOTE    : Numeric subject IDs only (no sub- prefix) for batch2.
          Metrics: FA, AD, MD, RD (configurable via METRICS).
          Matrix is symmetrized (mat + mat.T).
          Top 1-3 AAL region pairs are weighted by their percent overlap.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

METRICS = ["FA", "AD", "MD", "RD"]
# ────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_paths(subj):
    b = Path(BASE_DIR) / subj / "wm" / "derived"
    return {
        "pathstats": b / "pathstats" / f"{subj}_pathstats.csv",
        "aal_csv"  : b / "aal_summary" / f"{subj}_aal_summary.csv",
        "out_dir"  : b / "aal_summary" / "heatmap"
    }


def run_subject(subj):
    print(f"\n[{subj}]")
    p = get_paths(subj)

    for key in ("pathstats", "aal_csv"):
        if not p[key].exists():
            print(f"  ERROR: not found: {p[key]}")
            return False

    p["out_dir"].mkdir(parents=True, exist_ok=True)

    path_df = pd.read_csv(p["pathstats"])
    aal_df  = pd.read_csv(p["aal_csv"])

    for metric in METRICS:
        avg_col = f"{metric}_Avg"
        if avg_col not in path_df.columns:
            print(f"  WARN: {avg_col} not found, skipping")
            continue

        print(f"  {metric} ...", end=" ", flush=True)
        rows = []

        for _, p_row in path_df.iterrows():
            tract = p_row["tract"]
            tval  = p_row[avg_col]

            tract_aals = aal_df[aal_df["tract"] == tract]
            if tract_aals.empty:
                continue

            e1 = tract_aals[tract_aals["endpoint"].str.contains("endpt1")]
            e2 = tract_aals[tract_aals["endpoint"].str.contains("endpt2")]
            if e1.empty or e2.empty:
                continue

            for i in range(1, 4):
                for j in range(1, 4):
                    roi_u = e1[f"top{i}_aal_name"].values[0]
                    roi_v = e2[f"top{j}_aal_name"].values[0]
                    pct_u = e1[f"top{i}_percent"].values[0]
                    pct_v = e2[f"top{j}_percent"].values[0]

                    if pd.isna(roi_u) or pd.isna(roi_v):
                        continue
                    if roi_u in ("NA", "") or roi_v in ("NA", ""):
                        continue
                    if pd.isna(pct_u) or pd.isna(pct_v):
                        continue

                    w = (float(pct_u) / 100.0) * (float(pct_v) / 100.0)
                    rows.append({
                        "roi_u" : roi_u,
                        "roi_v" : roi_v,
                        "weight": tval * w,
                        "tract" : tract,
                    })

        if not rows:
            print("no edges")
            continue

        edge_df = pd.DataFrame(rows)
        mat_df  = edge_df.groupby(["roi_u", "roi_v"])["weight"].sum().reset_index()
        mat     = mat_df.pivot(index="roi_u", columns="roi_v", values="weight").fillna(0)
        mat     = mat.add(mat.T, fill_value=0)

        out = p["out_dir"]
        mat.to_csv(out / f"aal_aal_{metric}_matrix_{subj}.csv")
        edge_df.to_csv(out / f"aal_edge_details_{metric}_{subj}.csv", index=False)

        plt.figure(figsize=(12, 10))
        plt.imshow(mat.values, cmap="Greys", interpolation="nearest")
        plt.colorbar(label=metric)
        plt.title(f"AAL-AAL {metric} Connectivity Matrix ({subj})")
        plt.tight_layout()
        plt.savefig(out / f"aal_aal_{metric}_heatmap_{subj}.png", dpi=200)
        plt.close()

        print(f"OK ({len(edge_df)} edges)")

    return True


def main():
    print("=" * 52)
    print("  no8_aal_heatmap.py")
    print(f"  Subjects : {len(SUBJECTS)}")
    print(f"  Metrics  : {METRICS}")
    print("=" * 52)

    failed = []
    for subj in SUBJECTS:
        try:
            ok = run_subject(subj)
            if not ok:
                failed.append(subj)
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
