#!/usr/bin/env python3
"""
no7_aal_summary.py
====================
PURPOSE : For each tract endpoint (endpt1, endpt2), scan all active voxels
          in endpt.pd.nii.gz directly against the DWI-space AAL atlas
          (dwi_wsst_all_ROIs_AAL) and compute the top-3 AAL regions
          with their overlap percentages.
          Replaces no6~no9 (c1-matching pipeline) with direct DWI-space lookup.

PIPELINE CONTEXT:
  endpt{1,2}.pd.nii.gz  (DWI space, TRACULA output)
      ↓ active voxels (pd > 0) looked up in dwi_wsst_all_ROIs_AAL
      → AAL region id + name per voxel
      → top3 by voxel count + percent
  → aal_summary.csv  (same format as no9 output, drop-in replacement)

INPUT   : {BASE_DIR}/{subj}/wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz
          {BASE_DIR}/{subj}/gm/derived/
              dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii
          {LUT_PATH}  AAL3v1.nii.txt

OUTPUT  : {BASE_DIR}/{subj}/wm/derived/aal_summary/
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
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

LUT_PATH = (
    "/bml/projects/06_resilience/projects"
    "/06-12_wm-gm-fc-connectome/derivatives/template/AAL3/AAL3v1.nii.txt"
)
TOPN = 3          # number of top AAL regions to report
PD_THRESHOLD = 0  # voxels with pd > this value are counted (0 = any active voxel)
# ────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib


def load_lut(path):
    lut = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    lut[int(parts[0])] = " ".join(parts[1:])
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

    der_dir  = Path(BASE_DIR) / subj / "gm" / "derived"
    dpath    = Path(BASE_DIR) / subj / "wm" / "freesurfer" / subj / "dpath"
    out_dir  = Path(BASE_DIR) / subj / "wm" / "derived" / "aal_summary"

    aal_file = der_dir / f"dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii"

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
    print("=" * 52)
    print("  no7_aal_summary.py")
    print(f"  Subjects : {len(SUBJECTS)}")
    print("=" * 52)

    lut = load_lut(LUT_PATH)
    print(f"  LUT loaded: {len(lut)} AAL regions")

    failed = []
    for subj in SUBJECTS:
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
