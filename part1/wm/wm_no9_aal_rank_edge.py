#!/usr/bin/env python3
"""
no9_aal_rank_edge.py
=================
PURPOSE : Rank individual ROI-ROI edges by weighted connectivity score
          per WM metric, sorted descending.

PIPELINE CONTEXT:
  no8 → aal_edge_details_{metric}_{subj}.csv
      ↓ sort by weight descending
  → aal_edge_rank_{metric}_{subj}.csv

INPUT   : {BASE_DIR}/{subj}/wm/derived/aal_summary/heatmap/
              aal_edge_details_{metric}_{subj}.csv

OUTPUT  : {BASE_DIR}/{subj}/wm/derived/aal_summary/rank_edge/
              aal_edge_rank_{metric}_{subj}.csv
          Columns: roi_u, roi_v, weight, tract  (sorted by weight desc)

NOTE    : Numeric subject IDs only (no sub- prefix) for batch2.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

METRICS = ["FA", "AD", "MD", "RD"]
# ────────────────────────────────────────────────────────────

import pandas as pd
from pathlib import Path


def run_subject(subj):
    print(f"\n[{subj}]")

    in_dir  = Path(BASE_DIR) / subj / "wm" / "derived" / "aal_summary" / "heatmap"
    out_dir = Path(BASE_DIR) / subj / "wm" / "derived" / "aal_summary" / "rank_edge"

    if not in_dir.exists():
        print(f"  ERROR: heatmap dir not found: {in_dir}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        edge_csv = in_dir / f"aal_edge_details_{metric}_{subj}.csv"
        if not edge_csv.exists():
            print(f"  WARN: {edge_csv.name} not found, skipping")
            continue

        df = pd.read_csv(edge_csv)
        ranked = df.sort_values("weight", ascending=False).reset_index(drop=True)

        out_csv = out_dir / f"aal_edge_rank_{metric}_{subj}.csv"
        ranked.to_csv(out_csv, index=False)
        print(f"  {metric}: {len(ranked)} edges → {out_csv.name}")

    return True


def main():
    print("=" * 52)
    print("  no9_aal_rank_edge.py")
    print(f"  Subjects : {len(SUBJECTS)}")
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
