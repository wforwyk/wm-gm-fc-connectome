#!/usr/bin/env python3
# =============================================================================
# pt2_no2_wm_present_absent_fc.py
#
# PURPOSE:
#   For each subject, classify all AAL region pairs as WM-present or
#   WM-absent based on TRACULA tract endpoint AAL assignments, then
#   compute mean FC and mean geodesic distance per pair. For WM-present
#   pairs, attach the tract RD value.
#
# ANALYSIS 1 LOGIC:
#   WM-present pair  : at least one QC-passing TRACULA tract has endpt1 in
#                      region A and endpt2 in region B (or vice versa),
#                      determined from the subject's aal_summary.csv
#   WM-absent pair   : no such tract exists for that pair
#
#   For each pair (A, B):
#     - Collect all GM voxels in region A and region B from the bnb mask
#     - Extract their pairwise FC values from the upper triangle of R
#     - Extract their pairwise geodesic distances from the upper triangle of W_a
#     - Compute mean_FC, mean_dist_mm, n_pairs
#     - If WM-present: attach tract name and mean RD from pathstats
#
# INPUT FILES (per subject):
#   bnb mask   : gm/func/derived/bnbmdenoised_detrended_rest_{subj}.nii
#   FC matrix  : gm/func/derived/fc_results/{subj}_R.npz          (arr_0, n_gm x n_gm)
#   dist matrix: gm/anat/derived/dist_results/{subj}_W_a.npz      (arr_0, n_gm x n_gm)
#   AAL atlas  : gm/derived/fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-01_T1w_YOUR_DARTEL_TEMPLATE.nii
#   AAL summary: wm/derived/aal_summary/{subj}_aal_summary.csv
#   pathstats  : wm/derived/pathstats/{subj}_pathstats.csv
#
# REFERENCE FILE (atlas-level, from pt2_no01):
#   aal3_region_category.csv
#
# OUTPUT (per subject):
#   {subj}_wm_present_absent_pairs.csv
#     columns: subject, roi_id_a, roi_name_a, roi_id_b, roi_name_b,
#              category_a, category_b, wm_group, tract_name,
#              mean_FC, mean_dist_mm, n_voxel_pairs, mean_RD
#
# NOTES:
#   - Only pairs where both regions are include=1 (from region_category CSV)
#     are processed.
#   - R.npz holds raw Pearson r; absolute value is NOT taken here and no
#     Fisher-Z is applied, so raw r is preserved for slope analysis.
#     (pt2_no03 applies arctanh to the per-pair means written here.)
#   - Geodesic distance from W_a is in mm.
#   - R and W_a are dense (n_gm x n_gm). This script flattens their upper
#     triangles once per subject via np.triu_indices, then addresses pairs
#     by the closed-form position i*n - i*(i+1)//2 + j - i - 1.
#   - Voxel row order is the C-order of the non-zero voxels of the bnb mask,
#     which is the order gm_no6 used to build R and W_a.
#   - WM-present classification uses the top1 AAL id of each endpoint from
#     aal_summary.csv (pt1_wm_no07 also reports top2/top3, unused here).
#   - For WM-present pairs with multiple tracts connecting the same pair,
#     all tract rows are written separately (one row per tract).
#   - Subjects with missing bnb mask or FC file are skipped with a warning.
#   - QC-excluded tracts (rh.atr, acomm) are excluded from WM-present
#     classification.
#
# AUTHOR : BrainWorld pipeline
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from itertools import combinations

# =============================================================================
# USER SETTINGS
# =============================================================================
BATCH_DIR    = '/path/to/your/project/derivatives/batch'
TEMPLATE_DIR = '/path/to/your/project/derivatives/template/AAL3'
OUTPUT_DIR   = '/path/to/your/project/derivatives/pt2_group_results/no2_wm_fc'
TRACT_SUFFIX = '_avg16_syn_bbr'
SOURCE_SESSION = 'ses-01'
DARTEL_TEMPLATE_ID = 'YOUR_DARTEL_TEMPLATE'

# QC-excluded tracts (kept identical in pt2_no04 / no08)
QC_EXCLUDE   = {'rh.atr', 'acomm'}

# Subjects to process (fill in or use glob)
SUBJECT_LIST = sorted([
    # filter: numeric-only directory names (exclude output dirs, group results, etc.)
    os.path.basename(p)
    for p in glob.glob(os.path.join(BATCH_DIR, '*'))
    if os.path.isdir(p) and os.path.basename(p).isdigit()
])
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Load atlas-level reference tables
# --------------------------------------------------------------------------
region_df = pd.read_csv(os.path.join(TEMPLATE_DIR, 'aal3_region_category.csv'))
included_ids = set(region_df[region_df['include'] == 1]['roi_id'].tolist())
roi_to_name  = dict(zip(region_df['roi_id'], region_df['roi_name']))
roi_to_cat   = dict(zip(region_df['roi_id'], region_df['category']))

print(f"Included AAL regions: {len(included_ids)}")

# --------------------------------------------------------------------------
# Helper: get voxel indices in each AAL region from bnb mask + AAL mask
# --------------------------------------------------------------------------
def get_gm_voxel_indices_by_region(bnb_img, aal_img):
    """
    Returns dict: roi_id -> np.array of linear indices into the sorted
    non-zero voxel list of the bnb mask (0-based, matching R/W_a row order).
    """
    bnb_data = bnb_img.get_fdata()
    aal_data = aal_img.get_fdata()

    # GM voxel mask (non-zero in bnb, first volume if 4D)
    if bnb_data.ndim == 4:
        gm_mask = bnb_data[..., 0] > 0
    else:
        gm_mask = bnb_data > 0

    # flat linear indices of GM voxels (sorted, matching matrix row order)
    gm_flat_indices = np.where(gm_mask.ravel())[0]

    # AAL label for each GM voxel
    aal_flat = aal_data.ravel()[gm_flat_indices].astype(int)

    region_voxel_map = {}
    for roi_id in included_ids:
        mask_idx = np.where(aal_flat == roi_id)[0]  # index into gm_flat_indices
        if len(mask_idx) > 0:
            region_voxel_map[roi_id] = mask_idx

    return region_voxel_map

# --------------------------------------------------------------------------
# Helper: extract pairwise FC and distance for two sets of voxel indices
# --------------------------------------------------------------------------
def get_pair_stats(idx_a, idx_b, n_gm, vec_r, vec_w):
    """
    Given voxel index arrays idx_a and idx_b (into the n_gm x n_gm matrix),
    extract all cross-pair FC and distance values from upper-triangle vectors.
    Returns mean_FC, mean_dist, n_pairs.

    Fully vectorised: the cross product of the two index sets is built with
    repeat/tile rather than a Python double loop, matching the `positions`
    helper in pt2_no5 and the `upper` helper in pt2_no9.  With regions of a
    few hundred voxels each and thousands of region pairs per subject, the
    loop form was the dominant cost of this script.
    """
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    if idx_a.size == 0 or idx_b.size == 0:
        return np.nan, np.nan, 0

    # every (a, b) combination, ordered so that i < j (upper triangle)
    a = np.repeat(idx_a, idx_b.size)
    b = np.tile(idx_b, idx_a.size)
    i = np.minimum(a, b)
    j = np.maximum(a, b)

    # i == j drops voxels shared by both index sets; i < j also guarantees the
    # closed-form position below lands inside the upper-triangle vector.
    keep = i < j
    if not keep.any():
        return np.nan, np.nan, 0
    i, j = i[keep], j[keep]

    # (i, j) upper-triangle matrix indices -> 1D vector positions
    # np.triu_indices ordering: row-major upper triangle
    vec_pos = i * n_gm - i * (i + 1) // 2 + j - i - 1

    fc_vals   = vec_r[vec_pos].astype(float)
    dist_vals = vec_w[vec_pos].astype(float)
    dist_vals[~np.isfinite(dist_vals)] = np.nan  # exclude inf (disconnected GM pairs)

    return float(np.nanmean(fc_vals)), float(np.nanmean(dist_vals)), int(vec_pos.size)

# --------------------------------------------------------------------------
# Helper: load WM-present pairs for a subject from aal_summary
# --------------------------------------------------------------------------
def get_wm_present_pairs(subj, aal_summary_path, pathstats_path):
    """
    Returns list of dicts: {tract, roi_id_a, roi_id_b, mean_RD}
    Each entry = one tract connecting two AAL regions.
    """
    aal_df = pd.read_csv(aal_summary_path)

    # clean tract name (strip suffix)
    aal_df['tract_clean'] = aal_df['tract'].str.replace(TRACT_SUFFIX, '', regex=False)

    # exclude QC-excluded tracts
    aal_df = aal_df[~aal_df['tract_clean'].isin(QC_EXCLUDE)]

    # load RD from pathstats
    rd_map = {}
    if os.path.exists(pathstats_path):
        ps_df = pd.read_csv(pathstats_path)
        ps_df['tract_clean'] = ps_df['tract'].str.replace(TRACT_SUFFIX, '', regex=False)
        rd_col = next((c for c in ['RD_Avg', 'mean_RD', 'RD'] if c in ps_df.columns), None)
        if rd_col:
            for _, row in ps_df.iterrows():
                rd_map[row['tract_clean']] = row[rd_col]

    # build tract -> (endpt1_roi, endpt2_roi) mapping using top1 AAL id
    wm_pairs = []
    tracts = aal_df['tract_clean'].unique()
    for tract in tracts:
        tract_rows = aal_df[aal_df['tract_clean'] == tract]
        ep1_rows   = tract_rows[tract_rows['endpoint'] == 'endpt1']
        ep2_rows   = tract_rows[tract_rows['endpoint'] == 'endpt2']

        if ep1_rows.empty or ep2_rows.empty:
            continue

        val_a = ep1_rows.iloc[0]['top1_aal_id']
        val_b = ep2_rows.iloc[0]['top1_aal_id']
        if pd.isna(val_a) or pd.isna(val_b):
            continue  # skip tracts with missing AAL assignment

        roi_a = int(val_a)
        roi_b = int(val_b)

        if roi_a == roi_b:
            continue  # same region, skip

        pair = (min(roi_a, roi_b), max(roi_a, roi_b))
        wm_pairs.append({
            'tract'  : tract,
            'roi_id_a': pair[0],
            'roi_id_b': pair[1],
            'mean_RD' : rd_map.get(tract, np.nan)
        })

    return wm_pairs

# --------------------------------------------------------------------------
# Main subject loop
# --------------------------------------------------------------------------
failed_subjects = []

for subj in SUBJECT_LIST:
    print(f"\n--- Subject {subj} ---")

    # file paths
    bnb_path    = os.path.join(BATCH_DIR, subj, 'gm', 'func', 'derived',
                               f'bnbmdenoised_detrended_rest_{subj}.nii')
    fc_path     = os.path.join(BATCH_DIR, subj, 'gm', 'func', 'derived',
                               'fc_results', f'{subj}_R.npz')
    dist_path   = os.path.join(BATCH_DIR, subj, 'gm', 'anat', 'derived',
                               'dist_results', f'{subj}_W_a.npz')
    aal_path    = os.path.join(BATCH_DIR, subj, 'wm', 'derived', 'aal_summary',
                               f'{subj}_aal_summary.csv')
    ps_path     = os.path.join(BATCH_DIR, subj, 'wm', 'derived', 'pathstats',
                               f'{subj}_pathstats.csv')
    # AAL mask in fMRI space (warped, from no7 pipeline output)
    aal_mask_path = os.path.join(BATCH_DIR, subj, 'gm', 'derived',
                                 f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii')

    # check required files
    missing = [p for p in [bnb_path, fc_path, dist_path, aal_path, aal_mask_path]
               if not os.path.exists(p)]
    if missing:
        print(f"  SKIP: missing files:")
        for m in missing:
            print(f"    {m}")
        failed_subjects.append(subj)
        continue

    try:
        # load GM mask and AAL mask
        bnb_img = nib.load(bnb_path)
        aal_img = nib.load(aal_mask_path)

        bnb_data = bnb_img.get_fdata()
        if bnb_data.ndim == 4:
            gm_mask = bnb_data[..., 0] > 0
        else:
            gm_mask = bnb_data > 0
        n_gm = int(gm_mask.sum())
        print(f"  GM voxels: {n_gm}")

        # load FC vector and distance matrix
        vec_r = np.load(fc_path)['arr_0']
        # R.npz is dense (n_gm x n_gm); extract upper triangle as 1D
        triu_i, triu_j = np.triu_indices(n_gm, k=1)
        vec_r_1d = vec_r[triu_i, triu_j].astype(np.float32)
        del vec_r

        vec_w = np.load(dist_path)['arr_0']
        vec_w_1d = vec_w[triu_i, triu_j].astype(np.float32)
        del vec_w

        print(f"  FC vector length: {len(vec_r_1d)}")

        # get region -> voxel index mapping
        region_voxel_map = get_gm_voxel_indices_by_region(bnb_img, aal_img)
        regions_present  = set(region_voxel_map.keys())
        print(f"  AAL regions with GM voxels: {len(regions_present)}")

        # get WM-present pairs
        wm_pairs    = get_wm_present_pairs(subj, aal_path, ps_path)
        wm_pair_set = {(p['roi_id_a'], p['roi_id_b']) for p in wm_pairs}
        print(f"  WM-present pairs: {len(wm_pair_set)}")

        # generate all included region pairs
        included_present = sorted(included_ids & regions_present)
        all_pairs        = list(combinations(included_present, 2))
        print(f"  Total included pairs: {len(all_pairs)}")

        # compute FC and distance per pair
        output_rows = []

        for (roi_a, roi_b) in all_pairs:
            idx_a = region_voxel_map[roi_a]
            idx_b = region_voxel_map[roi_b]

            mean_fc, mean_dist, n_pairs = get_pair_stats(
                idx_a, idx_b, n_gm, vec_r_1d, vec_w_1d
            )

            pair_key = (min(roi_a, roi_b), max(roi_a, roi_b))

            if pair_key in wm_pair_set:
                # WM-present: one row per tract
                matching_tracts = [p for p in wm_pairs
                                   if (p['roi_id_a'], p['roi_id_b']) == pair_key]
                for tp in matching_tracts:
                    output_rows.append({
                        'subject'      : subj,
                        'roi_id_a'     : roi_a,
                        'roi_name_a'   : roi_to_name.get(roi_a, f'ROI_{roi_a}'),
                        'category_a'   : roi_to_cat.get(roi_a, 'unknown'),
                        'roi_id_b'     : roi_b,
                        'roi_name_b'   : roi_to_name.get(roi_b, f'ROI_{roi_b}'),
                        'category_b'   : roi_to_cat.get(roi_b, 'unknown'),
                        'wm_group'     : 'wm_present',
                        'tract_name'   : tp['tract'],
                        'mean_FC'      : mean_fc,
                        'mean_dist_mm' : mean_dist,
                        'n_voxel_pairs': n_pairs,
                        'mean_RD'      : tp['mean_RD']
                    })
            else:
                # WM-absent: single row
                output_rows.append({
                    'subject'      : subj,
                    'roi_id_a'     : roi_a,
                    'roi_name_a'   : roi_to_name.get(roi_a, f'ROI_{roi_a}'),
                    'category_a'   : roi_to_cat.get(roi_a, 'unknown'),
                    'roi_id_b'     : roi_b,
                    'roi_name_b'   : roi_to_name.get(roi_b, f'ROI_{roi_b}'),
                    'category_b'   : roi_to_cat.get(roi_b, 'unknown'),
                    'wm_group'     : 'wm_absent',
                    'tract_name'   : np.nan,
                    'mean_FC'      : mean_fc,
                    'mean_dist_mm' : mean_dist,
                    'n_voxel_pairs': n_pairs,
                    'mean_RD'      : np.nan
                })

        # save subject output
        out_df = pd.DataFrame(output_rows)
        out_path = os.path.join(OUTPUT_DIR, f'{subj}_wm_present_absent_pairs.csv')
        out_df.to_csv(out_path, index=False)

        n_present = (out_df['wm_group'] == 'wm_present').sum()
        n_absent  = (out_df['wm_group'] == 'wm_absent').sum()
        print(f"  WM-present rows: {n_present} | WM-absent rows: {n_absent}")
        print(f"  Saved: {out_path}")

    except Exception as e:
        print(f"  ERROR: {e}")
        failed_subjects.append(subj)
        continue

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "="*60)
print("DONE")
print(f"Failed subjects ({len(failed_subjects)}): {failed_subjects}")
