#!/usr/bin/env python3
# =============================================================================
# pt2_no03b_aim1_composition_sensitivity.py
#
# PURPOSE:
#   Sensitivity analysis for the Aim 1 primary contrast (pt2_no03).  The pooled
#   WM-present vs WM-absent comparison mixes region-pair categories: WM-present
#   pairs are mostly cortico-cortical, whereas WM-absent pairs include many
#   pairs with thalamic or subcortical-limbic regions.  Log-distance adjustment
#   does not remove this composition difference.  This script repeats the
#   Aim 1 contrast while holding the region-pair category constant.
#
#   pt2_no03 is NOT modified and its outputs are NOT overwritten.  This script
#   only reads two no03 outputs and writes to its own output directory.
#
# ANALYSES (participant is the unit of inference; one-sample t test per contrast):
#   A_all_pairs_replication
#       Original no03 primary contrast (log-distance only).  Must reproduce
#       the no03 result when participant and pair eligibility are identical.
#   B_cortico_cortical_only
#       Both groups restricted to pairs with both regions in category 'cortical';
#       FC ~ log(distance) refitted within these pairs, per participant.
#   C_category_adjusted
#       All pairs; FC ~ log(distance) + unordered region-category-pair dummies,
#       per participant.  Contrast = mean residual of WM-present minus WM-absent.
#   D_stratum_<category pair>
#       Contrast within each unordered category-pair stratum that contains
#       WM-present pairs (distance fit within the stratum).  Strata without
#       WM-present pairs (e.g. thalamic-thalamic) cannot be estimated and are
#       listed in the composition table.
#   Class-wise versions of B and C for the four pathway classes (WM-present
#   pairs of a class vs the WM-absent pool defined by the analysis).
#
# INPUT (written by pt2_no03_stats.py):
#   - pt2_no3_pair_level.csv        (deduplicated region pairs)
#   - pt2_no3_pooled_tract_level.csv (tract rows, for pathway classes)
#
# OUTPUT (new directory, nothing in no03_summary_stats is overwritten):
#   - pt2_no3b_composition_table.csv      pair counts by category pair x WM group
#   - pt2_no3b_subject_effects.csv        one row per participant x analysis
#   - pt2_no3b_summary_stats.csv / .txt   participant-level t tests
#   - pt2_no3b_noncortical_wm_present_regions.csv  non-cortical regions of WM-present pairs outside
#                                               cortico-cortical pairs (pair and participant counts)
#   - pt2_no3b_cc_descriptives.csv        per participant, cortico-cortical pairs:
#                                               pair counts, mean distance, slope, raw diff
#   - pt2_no3b_cc_descriptives_summary.csv  mean/SD of the above across participants
#   (family 'cc_raw' in the summary = unadjusted cortico-cortical contrasts, overall and by class)
#
# NOTES:
#   - 'cortico-cortical' = both regions have category 'cortical' (same
#     definition as is_cortical in no11/no12).
#     ACC subdivisions (acc_subdivision) and the AAL3 'subcortical_limbic' regions (hippocampus,
#     parahippocampal gyrus, amygdala) are therefore NOT counted as cortical.
#   - A participant enters a contrast only if it has at least MIN_PRESENT
#     WM-present and MIN_ABSENT WM-absent pairs in the relevant set.
#   - BH-FDR is applied within the families named in the 'family' column.
#
# USAGE (server):    python pt2_no03b_aim1_composition_sensitivity.py
# USAGE (override):  python pt2_no03b_aim1_composition_sensitivity.py INPUT_DIR OUTPUT_DIR
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from cohort_settings import filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR  = '/path/to/your/project/derivatives/results/part2/no03_summary_stats'
OUTPUT_DIR = '/path/to/your/project/derivatives/results/part2/no03b_composition_sensitivity'
MIN_PRESENT = 1
MIN_ABSENT = 3
MIN_SUBJECTS = 5
# =============================================================================

if len(sys.argv) == 3:
    INPUT_DIR, OUTPUT_DIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Must match PATHWAY_TYPE in pt2_no03_stats.py.
PATHWAY_TYPE = {
    'commissural pathways': ['acomm', 'cc.genu', 'cc.rostrum', 'cc.splenium', 'cc.bodyc',
                             'cc.bodyp', 'cc.bodypf', 'cc.bodypm', 'cc.bodyt'],
    'other pathways': ['lh.fx', 'rh.fx'],
    'projection pathways': ['lh.atr', 'rh.atr', 'lh.ar', 'rh.ar', 'lh.or', 'rh.or'],
    'association pathways': [f'{h}.{t}' for t in
                             ['af', 'slf1', 'slf2', 'slf3', 'uf', 'ilf', 'mlf', 'emc', 'fat', 'cbd', 'cbv']
                             for h in ('lh', 'rh')],
}
CLASS_ORDER = ['commissural pathways', 'projection pathways', 'association pathways', 'other pathways']


def assign_pathway_type(tract):
    for name, tracts in PATHWAY_TYPE.items():
        if tract in tracts:
            return name
    return 'other pathways'


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    valid = np.isfinite(pvals)
    if not valid.any():
        return out
    p = pvals[valid]
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty_like(adj)
    restored[order] = np.minimum(adj, 1.0)
    out[valid] = restored
    return out


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
pair_df = pd.read_csv(os.path.join(INPUT_DIR, 'pt2_no3_pair_level.csv'),
                      usecols=['subject', 'pair_id', 'wm_group_bin', 'category_a', 'category_b',
                               'roi_name_a', 'roi_name_b', 'mean_FC_z', 'mean_dist_mm'])
pair_df = filter_subject_rows(pair_df)
if pair_df.empty:
    raise ValueError('No selected participants in no03 pair table.')
pair_df = pair_df[np.isfinite(pair_df['mean_FC_z']) & np.isfinite(pair_df['mean_dist_mm'])
                  & (pair_df['mean_dist_mm'] > 0)].copy()
pair_df['log_dist'] = np.log(pair_df['mean_dist_mm'])
cat_lo = np.where(pair_df['category_a'] <= pair_df['category_b'], pair_df['category_a'], pair_df['category_b'])
cat_hi = np.where(pair_df['category_a'] <= pair_df['category_b'], pair_df['category_b'], pair_df['category_a'])
pair_df['category_pair'] = pd.Series(cat_lo, index=pair_df.index) + '|' + pd.Series(cat_hi, index=pair_df.index)
pair_df['is_cc'] = (pair_df['category_a'] == 'cortical') & (pair_df['category_b'] == 'cortical')

tract_df = pd.read_csv(os.path.join(INPUT_DIR, 'pt2_no3_pooled_tract_level.csv'),
                       usecols=['subject', 'pair_id', 'wm_group', 'tract_name'])
tract_df = filter_subject_rows(tract_df)
tract_df = tract_df[tract_df['wm_group'] == 'wm_present'].copy()
tract_df['pathway_type'] = tract_df['tract_name'].apply(assign_pathway_type)
class_pairs = tract_df[['subject', 'pair_id', 'pathway_type']].drop_duplicates()

# Composition table (descriptive).
comp = (pair_df.groupby(['category_pair', 'wm_group_bin']).size().unstack(fill_value=0)
        .rename(columns={0: 'n_wm_absent_pairs', 1: 'n_wm_present_pairs'}).reset_index())
for col in ('n_wm_absent_pairs', 'n_wm_present_pairs'):
    if col not in comp:
        comp[col] = 0
comp['pct_of_wm_present'] = 100 * comp['n_wm_present_pairs'] / comp['n_wm_present_pairs'].sum()
comp['pct_of_wm_absent'] = 100 * comp['n_wm_absent_pairs'] / comp['n_wm_absent_pairs'].sum()
comp.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_composition_table.csv'), index=False)
print(comp.to_string(index=False))

# Which non-cortical regions the WM-present pairs involve (descriptive; one row per region).
wp = pair_df[(pair_df['wm_group_bin'] == 1) & ~pair_df['is_cc']].copy()
a_nc = wp['category_a'] != 'cortical'
wp['noncortical_region'] = np.where(a_nc, wp['roi_name_a'], wp['roi_name_b'])
wp['noncortical_category'] = np.where(a_nc, wp['category_a'], wp['category_b'])
region_tab = (wp.groupby(['noncortical_category', 'noncortical_region'])
              .agg(n_wm_present_pairs=('pair_id', 'size'), n_participants=('subject', 'nunique'))
              .reset_index().sort_values('n_wm_present_pairs', ascending=False))
region_tab['pct_of_noncortical_wm_present_pairs'] = 100 * region_tab['n_wm_present_pairs'] / region_tab['n_wm_present_pairs'].sum()
region_tab.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_noncortical_wm_present_regions.csv'), index=False)

# Class membership of each deduplicated WM-present pair row (a pair can belong to >1 class).
present_rows = pair_df[pair_df['wm_group_bin'] == 1][['subject', 'pair_id']]
class_member = {c: class_pairs[class_pairs['pathway_type'] == c][['subject', 'pair_id']]
                .assign(_m=True) for c in CLASS_ORDER}


# --------------------------------------------------------------------------
# 2. Participant-level contrasts
# --------------------------------------------------------------------------
def residuals(grp, with_category):
    cols = [np.ones(len(grp)), grp['log_dist'].to_numpy()]
    if with_category:
        dummies = pd.get_dummies(grp['category_pair'], drop_first=True).to_numpy(dtype=float)
        if dummies.shape[1]:
            cols.extend(dummies.T)
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, grp['mean_FC_z'].to_numpy(), rcond=None)
    return grp['mean_FC_z'].to_numpy() - X @ beta


def contrast(grp, with_category, class_mask=None):
    """Mean residual of WM-present (optionally one class) minus mean residual of WM-absent."""
    present = (grp['wm_group_bin'] == 1).to_numpy()
    absent = ~present
    if absent.sum() < MIN_ABSENT or present.sum() < MIN_PRESENT or len(grp) < 5:
        return np.nan
    res = residuals(grp, with_category)
    sel = present if class_mask is None else (present & class_mask)
    if sel.sum() < MIN_PRESENT:
        return np.nan
    return res[sel].mean() - res[absent].mean()


ANALYSES = {
    # label: (pair filter, with_category_dummies, family)
    'A_all_pairs_replication':  (lambda g: g, False, 'primary'),
    'B_cortico_cortical_only':  (lambda g: g[g['is_cc']], False, 'primary'),
    'C_category_adjusted':      (lambda g: g, True, 'primary'),
}
strata = sorted(comp.loc[comp['n_wm_present_pairs'] > 0, 'category_pair'])
for s in strata:
    ANALYSES[f'D_stratum_{s}'] = ((lambda g, s=s: g[g['category_pair'] == s]), False, 'strata')

rows = []
desc_rows = []
for subj, grp_all in pair_df.groupby('subject'):
    members = {c: (grp_all.merge(class_member[c], on=['subject', 'pair_id'], how='left')['_m']
                   .fillna(False).to_numpy(dtype=bool)) for c in CLASS_ORDER}
    grp_all = grp_all.reset_index(drop=True)
    for label, (filt, with_cat, family) in ANALYSES.items():
        sub = filt(grp_all)
        rows.append({'subject': subj, 'analysis': label, 'class': 'all', 'family': family,
                     'diff': contrast(sub, with_cat)})
    # class-wise: B (cortico-cortical pool) and C (category-adjusted, all pairs)
    for c in CLASS_ORDER:
        mask_all = members[c]
        cc_mask = grp_all['is_cc'].to_numpy()
        sub_cc = grp_all[cc_mask]
        rows.append({'subject': subj, 'analysis': 'B_cortico_cortical_only', 'class': c,
                     'family': 'class_B',
                     'diff': contrast(sub_cc, False, class_mask=mask_all[cc_mask])})
        rows.append({'subject': subj, 'analysis': 'C_category_adjusted', 'class': c,
                     'family': 'class_C',
                     'diff': contrast(grp_all, True, class_mask=mask_all)})
    # reference: original no03 class-wise estimate (log-distance only, all pairs)
    for c in CLASS_ORDER:
        rows.append({'subject': subj, 'analysis': 'A_all_pairs_replication', 'class': c,
                     'family': 'class_A', 'diff': contrast(grp_all, False, class_mask=members[c])})

    # descriptives and unadjusted contrasts for the cortico-cortical pairs (family 'cc_raw')
    cc_mask = grp_all['is_cc'].to_numpy()
    sub_cc = grp_all[cc_mask]
    pres = (sub_cc['wm_group_bin'] == 1).to_numpy()
    if pres.sum() >= MIN_PRESENT and (~pres).sum() >= MIN_ABSENT:
        slope = np.polyfit(sub_cc['log_dist'], sub_cc['mean_FC_z'], 1)[0]
        fc = sub_cc['mean_FC_z'].to_numpy()
        desc_rows.append({'subject': subj, 'n_pairs': len(sub_cc), 'n_wm_present': int(pres.sum()),
                          'n_wm_absent': int((~pres).sum()),
                          'mean_dist_present_mm': sub_cc.loc[pres, 'mean_dist_mm'].mean(),
                          'mean_dist_absent_mm': sub_cc.loc[~pres, 'mean_dist_mm'].mean(),
                          'log_distance_slope': slope,
                          'raw_diff': fc[pres].mean() - fc[~pres].mean()})
        rows.append({'subject': subj, 'analysis': 'B_raw_unadjusted_cortico_cortical', 'class': 'all',
                     'family': 'cc_raw', 'diff': fc[pres].mean() - fc[~pres].mean()})
        for c in CLASS_ORDER:
            sel = members[c][cc_mask] & pres
            rows.append({'subject': subj, 'analysis': 'B_raw_unadjusted_cortico_cortical', 'class': c,
                         'family': 'cc_raw',
                         'diff': (fc[sel].mean() - fc[~pres].mean()) if sel.sum() >= MIN_PRESENT else np.nan})

eff = pd.DataFrame(rows)
desc = pd.DataFrame(desc_rows, columns=['subject', 'n_pairs', 'n_wm_present', 'n_wm_absent',
    'mean_dist_present_mm', 'mean_dist_absent_mm', 'log_distance_slope', 'raw_diff'])
desc.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_cc_descriptives.csv'), index=False)
desc_summary = desc.drop(columns='subject').agg(['count', 'mean', 'std']).T
desc_summary.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_cc_descriptives_summary.csv'))
print('\nCortico-cortical descriptives across participants:\n' + desc_summary.to_string())
eff.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_subject_effects.csv'), index=False)

# --------------------------------------------------------------------------
# 3. Participant-level tests
# --------------------------------------------------------------------------
summary = []
for (analysis, cls, family), g in eff.groupby(['analysis', 'class', 'family'], sort=False):
    v = g['diff'].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    rec = {'analysis': analysis, 'class': cls, 'family': family, 'n_subjects': len(v), 'p': np.nan}
    if len(v) >= MIN_SUBJECTS:
        t, p = stats.ttest_1samp(v, 0)
        se = v.std(ddof=1) / np.sqrt(len(v))
        half = stats.t.ppf(.975, len(v) - 1) * se
        rec.update({'mean': v.mean(), 'sd': v.std(ddof=1), 'ci95_lo': v.mean() - half,
                    'ci95_hi': v.mean() + half, 't': t, 'df': len(v) - 1, 'p': p,
                    'd_z': v.mean() / v.std(ddof=1)})
    summary.append(rec)
summary = pd.DataFrame(summary)
summary['q_bh_within_family'] = np.nan
for fam, idx in summary.groupby('family').groups.items():
    summary.loc[idx, 'q_bh_within_family'] = benjamini_hochberg(summary.loc[idx, 'p'])
summary.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3b_summary_stats.csv'), index=False)

with open(os.path.join(OUTPUT_DIR, 'pt2_no3b_summary_stats.txt'), 'w') as f:
    f.write('Aim 1 composition sensitivity (Fisher z; participant-level one-sample t tests)\n')
    f.write('Compare A_all_pairs_replication with no03 using identical participant and pair eligibility.\n')
    f.write('Strata without WM-present pairs cannot be estimated (see composition table).\n\n')
    f.write(summary.to_string(index=False))
    f.write('\n\nComposition (region-category pair x WM group):\n')
    f.write(comp.to_string(index=False))

print('\n' + summary.to_string(index=False))
print('\nDone.')
