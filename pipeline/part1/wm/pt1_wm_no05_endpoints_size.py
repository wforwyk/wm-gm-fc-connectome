#!/usr/bin/env python3
"""
pt1_wm_no05_endpoints_size.py
==================
PURPOSE : Count active voxels in each tract endpoint probability density map
          (endpt.pd.nii.gz) directly from TRACULA output.
          Voxel count serves as a proxy for tract endpoint size,
          useful for normalization and QC filtering.

PIPELINE CONTEXT:
  endpt{1,2}.pd.nii.gz  (DWI space, TRACULA output)
      ↓ count voxels with pd > 0
  → {subj}_voxelcount.csv  (tract × endpoint × n_voxels)

INPUT   : {BATCH_DIR}/{subj}/wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz

OUTPUT  : {BATCH_DIR}/{subj}/wm/derived/endpoints/{subj}_endpoints_size.csv
          Columns: subject, tract, endpoint, n_voxels

NOTE    :
          n_voxels reflects raw endpoint size before any thresholding.

"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "0001", "0002"
] # subject id must be in " " and divided by ,
SUBJECTS_TXT = ''
EXCLUDE_SUBJECTS = []


BATCH_DIR = ("/path/to/your/project/derivatives/batch")

PD_THRESHOLD = 0  # voxels with pd > this value are counted
# ────────────────────────────────────────────────────────────

import os
from pathlib import Path
import nibabel as nib
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subject_selection import resolve_subjects


def run_subject(subj):
    print(f"\n[{subj}]")

    dpath   = Path(BATCH_DIR) / subj / "wm" / "freesurfer" / subj / "dpath"
    out_dir = Path(BATCH_DIR) / subj / "wm" / "derived" / "endpoints"

    if not dpath.exists():
        print(f"  ERROR: dpath not found: {dpath}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    tract_dirs = sorted([d for d in dpath.iterdir() if d.is_dir()])
    print(f"  Tracts found: {len(tract_dirs)}")

    for tract_dir in tract_dirs:
        tract_name = tract_dir.name
        for ep_num in ["endpt1", "endpt2"]:
            ep_file = tract_dir / f"{ep_num}.pd.nii.gz"
            if not ep_file.exists():
                continue

            data  = nib.load(str(ep_file)).get_fdata()
            count = int((data > PD_THRESHOLD).sum())

            rows.append({
                "subject" : subj,
                "tract"   : tract_name,
                "endpoint": ep_num,
                "n_voxels": count,
            })

    if not rows:
        print(f"  WARNING: no endpoint files found")
        return False

    df = pd.DataFrame(rows)
    out_csv = out_dir / f"{subj}_endpoints_size.csv"
    df.to_csv(out_csv, index=False)
    print(f"  OK: {len(df)} rows → {out_csv}")
    return True


def main():
    subjects = resolve_subjects(SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, BATCH_DIR)
    print("=" * 52)
    print("  pt1_wm_no05_endpoints_size.py")
    print(f"  Subjects : {len(subjects)}")
    print("=" * 52)

    failed = []
    for subj in subjects:
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
