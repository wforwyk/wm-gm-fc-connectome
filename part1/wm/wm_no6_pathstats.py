#!/usr/bin/env python3
"""
no6_pathstats.py
=================
PURPOSE : Parse pathstats.overall.txt for each tract and aggregate
          into a single CSV per subject (FA, MD, AD, RD, etc.).

PIPELINE CONTEXT:
  {subj}/wm/freesurfer/{subj}/dpath/{tract}/pathstats.overall.txt
      ↓ parse key-value pairs
  → {subj}_pathstats.csv  (tract × WM metrics)

INPUT   : {BASE_DIR}/{subj}/wm/freesurfer/{subj}/dpath/{tract}/pathstats.overall.txt

OUTPUT  : {BASE_DIR}/{subj}/wm/derived/pathstats/{subj}_pathstats.csv
          Columns: tract, NumStreamlines, FA_Avg, MD_Avg, AD_Avg, RD_Avg, ...

NOTE    : Numeric subject IDs only (no sub- prefix) for batch2.
          Lines starting with '#' are skipped.
          Values are cast to float where possible.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

VERBOSE = True
# ────────────────────────────────────────────────────────────

from pathlib import Path
import pandas as pd


def parse_pathstats(file_path):
    stats = {}
    try:
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        stats[parts[0]] = float(parts[1])
                    except ValueError:
                        stats[parts[0]] = parts[1]
    except Exception as e:
        print(f"  ERROR reading {file_path.name}: {e}")
        return None
    return stats


def run_subject(subj):
    print(f"\n[{subj}]")

    dpath   = Path(BASE_DIR) / subj / "wm" / "freesurfer" / subj / "dpath"
    out_dir = Path(BASE_DIR) / subj / "wm" / "derived" / "pathstats"

    if not dpath.exists():
        print(f"  ERROR: dpath not found: {dpath}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tract_dirs = sorted([p for p in dpath.iterdir() if p.is_dir()])

    for tdir in tract_dirs:
        stats_path = tdir / "pathstats.overall.txt"
        if not stats_path.exists():
            continue
        stats = parse_pathstats(stats_path)
        if stats:
            row = {"tract": tdir.name}
            row.update(stats)
            rows.append(row)
            if VERBOSE:
                print(f"  - {tdir.name:40s} OK")

    if not rows:
        print(f"  WARNING: no pathstats found")
        return False

    df = pd.DataFrame(rows)
    cols = ["tract"] + [c for c in df.columns if c != "tract"]
    df = df[cols]

    out_csv = out_dir / f"{subj}_pathstats.csv"
    df.to_csv(out_csv, index=False)
    print(f"  OK: {len(df)} tracts → {out_csv}")
    return True


def main():
    print("=" * 52)
    print("  no6_pathstats.py")
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
