#!/usr/bin/env python3
# =============================================================================
# pt2_no1_region_category.py
#
# PURPOSE:
#   Generate a lookup table assigning each AAL3v1 region a category label
#   and an include/exclude flag for downstream FC analyses.
#
# INPUT:
#   - AAL3v1.nii.txt           : AAL3v1 region list, whitespace-separated
#                                "<id> <name> [<index>]". Most rows carry a
#                                trailing index column; a few (35/36
#                                Cingulate_Ant, 81/82 Thalamus) do not, so only
#                                field 2 is the name. pt1_wm_no07 / no11 read
#                                the same file with the same rule, which is what
#                                makes region names join across Part 1 and 2.
#   - ROI_MNI_V7_1mm_Border.mat: border voxel coordinates per region,
#                                used to derive physically adjacent region pairs
#
# OUTPUT:
#   - aal3_region_category.csv : per-region category and include flag
#   - aal3_adjacent_pairs.csv  : all physically adjacent AAL region pairs
#                                (derived from 6-connectivity of border voxels)
#
# CATEGORY LABELS:
#   cortical            : neocortical regions (ROI 1-74, excl. subcortical)
#   subcortical_limbic  : hippocampus, parahippocampal, amygdala (41-46)
#                         included: appear as TRACULA endpoints (fx, cbv)
#   subcortical_basal   : caudate, putamen, pallidum (75-80)
#                         excluded: not primary targets of GM FC analysis
#   thalamus_generic    : thalamus L/R (81-82)
#                         included: confirmed endpoint of lh.ar / rh.ar
#   thalamic_nuclei     : subdivided thalamic nuclei (121-150)
#                         included: confirmed endpoint of lh.atr (Thal_IL, Thal_LP)
#   acc_subdivision     : ACC subdivisions (151-156)
#                         included with flag: low frequency, monitor
#   cerebellum          : cerebellum + vermis (95-120)
#                         excluded: outside cortical FC analysis scope
#   brainstem           : VTA, SN, Red N, LC, Raphe (159-170)
#                         excluded: noise-level endpoint frequency
#   n_acc               : nucleus accumbens (157-158)
#                         excluded
#
# NOTES:
#   - ROI 35/36 (Cingulate_Ant_L/R) and 81/82 (Thalamus_L/R) are confirmed
#     active in AAL3v1 (appear in subject endpoint data); they are NOT replaced
#     by ACC subdivisions / thalamic nuclei in this atlas version.
#   - Adjacent pair detection uses 6-connectivity (face-touching only) on
#     border voxels in MNI space. Total pairs expected: ~600.
#   - This script runs once at the atlas level; output CSVs are used as
#     reference tables by all subsequent subject-level scripts.
#
# AUTHOR : BrainWorld pipeline
# =============================================================================

import os
import numpy as np
import pandas as pd
import scipy.io

# =============================================================================
# USER SETTINGS
# =============================================================================
AAL_TXT      = '/path/to/your/project/derivatives/template/AAL3/AAL3v1.nii.txt'
BORDER_MAT   = '/path/to/your/project/derivatives/template/AAL3/ROI_MNI_V7_1mm_Border.mat'
OUTPUT_DIR   = '/path/to/your/project/derivatives/template/AAL3'
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load AAL region list
# --------------------------------------------------------------------------
rows = []
with open(AAL_TXT, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            roi_id   = int(parts[0])
            roi_name = parts[1]
            rows.append({'roi_id': roi_id, 'roi_name': roi_name})

df = pd.DataFrame(rows)
print(f"Loaded {len(df)} AAL regions")

# --------------------------------------------------------------------------
# 2. Assign category and include flag
# --------------------------------------------------------------------------
def assign_category(roi_id):
    if roi_id in range(41, 47):
        return 'subcortical_limbic'
    elif roi_id in range(75, 81):
        return 'subcortical_basal'
    elif roi_id in [81, 82]:
        return 'thalamus_generic'
    elif roi_id in range(83, 95):
        return 'cortical'
    elif roi_id in range(1, 41):
        return 'cortical'
    elif roi_id in range(47, 75):
        return 'cortical'
    elif roi_id in range(95, 121):
        return 'cerebellum'
    elif roi_id in range(121, 151):
        return 'thalamic_nuclei'
    elif roi_id in range(151, 157):
        return 'acc_subdivision'
    elif roi_id in [157, 158]:
        return 'n_acc'
    elif roi_id in range(159, 171):
        return 'brainstem'
    else:
        return 'unknown'

INCLUDE_CATEGORIES = {
    'cortical', 'subcortical_limbic', 'thalamus_generic', 'thalamic_nuclei', 'acc_subdivision'
}

df['category'] = df['roi_id'].apply(assign_category)
df['include']  = df['category'].apply(lambda c: 1 if c in INCLUDE_CATEGORIES else 0)

# --------------------------------------------------------------------------
# 3. Print summary
# --------------------------------------------------------------------------
print("\n=== Region category summary ===")
summary = df.groupby(['category', 'include']).size().reset_index(name='count')
for _, row in summary.iterrows():
    flag = 'INCLUDE' if row['include'] else 'EXCLUDE'
    print(f"  {row['category']:<25s} {flag}  n={row['count']}")

total_include = df['include'].sum()
print(f"\nTotal included regions : {total_include}")
print(f"Total excluded regions : {len(df) - total_include}")

# --------------------------------------------------------------------------
# 4. Save region category CSV
# --------------------------------------------------------------------------
out_region = os.path.join(OUTPUT_DIR, 'aal3_region_category.csv')
df.to_csv(out_region, index=False)
print(f"\nSaved: {out_region}")

# --------------------------------------------------------------------------
# 5. Derive adjacent pairs from border voxel 6-connectivity
# --------------------------------------------------------------------------
print("\nLoading border voxel data...")
mat       = scipy.io.loadmat(BORDER_MAT)
BORDER_V   = mat['BORDER_V'].flatten().astype(int)
BORDER_XYZ = mat['BORDER_XYZ'].T         # (N, 3)

print(f"  border voxels: {len(BORDER_V)}")
print(f"  unique regions in border: {len(np.unique(BORDER_V))}")

# build coordinate -> roi lookup
coord_to_roi = {}
for i in range(len(BORDER_V)):
    key = (int(BORDER_XYZ[i, 0]), int(BORDER_XYZ[i, 1]), int(BORDER_XYZ[i, 2]))
    coord_to_roi[key] = BORDER_V[i]

# 6-connectivity adjacency search
directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
adjacent_pairs = set()

for i in range(len(BORDER_V)):
    roi_a = BORDER_V[i]
    x, y, z = int(BORDER_XYZ[i, 0]), int(BORDER_XYZ[i, 1]), int(BORDER_XYZ[i, 2])
    for dx, dy, dz in directions:
        neighbor = (x+dx, y+dy, z+dz)
        if neighbor in coord_to_roi:
            roi_b = coord_to_roi[neighbor]
            if roi_a != roi_b:
                adjacent_pairs.add((min(roi_a, roi_b), max(roi_a, roi_b)))

print(f"  total adjacent pairs found: {len(adjacent_pairs)}")

# build adjacency dataframe with category and include flags
roi_to_name     = dict(zip(df['roi_id'], df['roi_name']))
roi_to_cat      = dict(zip(df['roi_id'], df['category']))
roi_to_include  = dict(zip(df['roi_id'], df['include']))

adj_rows = []
for (a, b) in sorted(adjacent_pairs):
    adj_rows.append({
        'roi_id_a'   : a,
        'roi_name_a' : roi_to_name.get(a, f'ROI_{a}'),
        'category_a' : roi_to_cat.get(a, 'unknown'),
        'include_a'  : roi_to_include.get(a, 0),
        'roi_id_b'   : b,
        'roi_name_b' : roi_to_name.get(b, f'ROI_{b}'),
        'category_b' : roi_to_cat.get(b, 'unknown'),
        'include_b'  : roi_to_include.get(b, 0),
        'both_included': 1 if (roi_to_include.get(a, 0) == 1 and
                                roi_to_include.get(b, 0) == 1) else 0
    })

adj_df = pd.DataFrame(adj_rows)

out_adj = os.path.join(OUTPUT_DIR, 'aal3_adjacent_pairs.csv')
adj_df.to_csv(out_adj, index=False)
print(f"Saved: {out_adj}")

# summary
both_in = adj_df['both_included'].sum()
print(f"\nAdjacent pairs (both included): {both_in}")
print(f"Adjacent pairs (at least one excluded): {len(adj_df) - both_in}")

print("\nDone.")
