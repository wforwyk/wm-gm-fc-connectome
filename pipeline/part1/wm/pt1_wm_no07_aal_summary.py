#!/usr/bin/env python3
"""
pt1_wm_no07_aal_summary.py
===========================
PURPOSE : For each tract endpoint (endpt1, endpt2), scan all active voxels
          in endpt.pd.nii.gz directly against the DWI-space AAL atlas
          (dwi_wsst_all_ROIs_AAL) and compute the top-3 AAL regions
          with their overlap percentages.

PIPELINE CONTEXT:
  endpt{1,2}.pd.nii.gz  (DWI space, TRACULA output)
      ↓ active voxels (pd > 0) looked up in dwi_wsst_all_ROIs_AAL
      → AAL region id + name per voxel
      → top3 by voxel count + percent
  → {subj}_aal_summary.csv

INPUT   : {BATCH_DIR}/{subj}/wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz
          {BATCH_DIR}/{subj}/gm/derived/
              dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-01_T1w_YOUR_DARTEL_TEMPLATE.nii
          {AAL_LUT}  all_ROIs_AAL.txt

OUTPUT  : {BATCH_DIR}/{subj}/wm/derived/aal_summary/
              {subj}_aal_summary.csv
          Columns: tract, endpoint,
                   top1_aal_id, top1_aal_name, top1_percent,
                   top2_aal_id, top2_aal_name, top2_percent,
                   top3_aal_id, top3_aal_name, top3_percent

NOTE    :
          endpt.pd.nii.gz voxels with value > 0 are treated as active.
          AAL id = 0 (background) is excluded from region counting.
          If fewer than 3 regions found, remaining top slots are left empty.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = ["0001", "0002"]  # Numeric derivative IDs; replace with yours.
SUBJECTS_TXT = ''
EXCLUDE_SUBJECTS = []


BATCH_DIR = ("/path/to/your/project/derivatives/batch")

AAL_LUT = "/path/to/your/project/derivatives/template/DSIstudio/all_ROIs_AAL.txt"
SOURCE_SESSION = "ses-01"
DARTEL_TEMPLATE_ID = "YOUR_DARTEL_TEMPLATE"
TOPN = 3          # number of top AAL regions to report
PD_THRESHOLD = 0  # voxels with pd > this value are counted (0 = any active voxel)
# ────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subject_selection import resolve_subjects


def load_lut(path):
    """
    Read the paired DSI Studio export LUT: "<export_id> <ROI_name>".
    Optional trailing columns are ignored; canonical AAL3 IDs are not used here.
    """
    lut = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    lut[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    return lut


def top_aal_regions(ep_data, dwi_aal, lut, topn=3):
    """
    Given a 3D endpoint probability density array and a DWI-space AAL atlas,
    return the top-N AAL regions by voxel count with their percentages.
    Returns a list of dicts: [{aal_id, aal_name, percent}, ...]
    """
    active_ijk = np.argwhere(ep_data > PD_THRESHOLD)
    if len(active_ijk) == 0:
        return []

    sh = dwi_aal.shape
    aal_ids = []
    for i, j, k in active_ijk:
        if 0 <= i < sh[0] and 0 <= j < sh[1] and 0 <= k < sh[2]:
            rid = int(dwi_aal[i, j, k])
            if rid > 0:
                aal_ids.append(rid)

    if not aal_ids:
        return []

    total = len(aal_ids)
    unique, counts = np.unique(aal_ids, return_counts=True)
    order = np.argsort(-counts)  # descending

    results = []
    for idx in order[:topn]:
        rid = int(unique[idx])
        cnt = int(counts[idx])
        results.append({
            "aal_id"  : rid,
            "aal_name": lut.get(rid, "NA"),
            "percent" : round(100.0 * cnt / total, 2),
        })
    return results


def run_subject(subj, lut):
    print(f"\n[{subj}]")

    der_dir  = Path(BATCH_DIR) / subj / "gm" / "derived"
    dpath    = Path(BATCH_DIR) / subj / "wm" / "freesurfer" / subj / "dpath"
    out_dir  = Path(BATCH_DIR) / subj / "wm" / "derived" / "aal_summary"

    aal_file = der_dir / f"dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii"

    # ── check inputs ──────────────────────────────────────────
    if not aal_file.exists():
        print(f"  ERROR: DWI AAL atlas not found: {aal_file}")
        return False
    if not dpath.exists():
        print(f"  ERROR: dpath not found: {dpath}")
        return False

    dwi_aal = nib.load(str(aal_file)).get_fdata().astype(np.int32)
    print(f"  AAL atlas shape: {dwi_aal.shape}")

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

            ep_data = nib.load(str(ep_file)).get_fdata()
            top = top_aal_regions(ep_data, dwi_aal, lut, topn=TOPN)

            row = {"tract": tract_name, "endpoint": ep_num}
            for rank, r in enumerate(top, start=1):
                row[f"top{rank}_aal_id"]   = r["aal_id"]
                row[f"top{rank}_aal_name"] = r["aal_name"]
                row[f"top{rank}_percent"]  = r["percent"]
            # fill empty slots if fewer than TOPN regions found
            for rank in range(len(top) + 1, TOPN + 1):
                row[f"top{rank}_aal_id"]   = ""
                row[f"top{rank}_aal_name"] = ""
                row[f"top{rank}_percent"]  = ""

            rows.append(row)

    if not rows:
        print(f"  WARNING: no endpoint data found")
        return False

    cols = ["tract", "endpoint"]
    for rank in range(1, TOPN + 1):
        cols += [f"top{rank}_aal_id", f"top{rank}_aal_name", f"top{rank}_percent"]

    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values(["tract", "endpoint"]).reset_index(drop=True)

    out_csv = out_dir / f"{subj}_aal_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"  OK: {len(df)} rows → {out_csv}")
    return True


def main():
    subjects = resolve_subjects(SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, BATCH_DIR)
    print("=" * 52)
    print("  pt1_wm_no07_aal_summary.py")
    print(f"  Subjects : {len(subjects)}")
    print("=" * 52)

    lut = load_lut(AAL_LUT)
    print(f"  AAL_LUT loaded: {len(lut)} AAL regions")

    failed = []
    for subj in subjects:
        try:
            ok = run_subject(subj, lut)
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
