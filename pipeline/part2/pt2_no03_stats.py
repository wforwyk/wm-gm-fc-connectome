#!/usr/bin/env python3
# =============================================================================
# pt2_no03_stats.py
#
# PURPOSE:
#   Aggregate pt2_no2 per-subject CSVs and test whether WM-present pairs
#   show higher FC than WM-absent pairs after subject-specific adjustment for
#   log geodesic distance.  The raw contrast is retained as descriptive;
#   tract-level rows are retained only for tract/RD analyses.
#
# KEY DESIGN:
#   WM group (present/absent) is a WITHIN-subject factor. Every subject has
#   both WM-present and WM-absent pairs. Statistical tests are therefore
#   conducted at the subject level:
#     Main test  : per-subject mean residual-FC diff (present - absent) after
#                  subject-specific FC~log(distance) residualization, tested
#                  against zero via one-sample t-test (df = n_subj - 1)
#     System test: distance-adjusted, deduplicated within each of four pathway
#                  classes: commissural, projection, association, other (fornix).
#     Optional RD screen: disabled by default; descriptive, not myelin-specific.
#
# INPUT:
#   - pt2_no2 output CSVs: {subj}_wm_present_absent_pairs.csv
#   - aal3_region_category.csv (for cortical filter)
#
# OUTPUT:
#   - pt2_no3_pair_level.csv / pt2_no3_pooled_tract_level.csv
#   - pt2_no3_distance_adjusted_stats.csv / pt2_no3_subject_effects.csv
#   - pt2_no3_within_subj_stats.txt  : primary distance-adjusted t-test
#   - pt2_no3_system_stats.txt       : per-system within-subject t-test
#   - exploratory_rd_fc/             : optional descriptive RD-FC outputs
#   - pt2_no3_fig_violin.png         : FC distribution by WM group
#   - pt2_no3_fig_scatter.png        : FC ~ distance (log fit, red vs blue)
#   - pt2_no3_fig_fc_rd.png          : FC ~ RD (exp decay, per-system fits)
#   - pt2_no3_fig_system.png         : within-subject FC diff by tract system
#
# NOTES:
#   - Included pairs only (both regions include==1 from aal3_region_category.csv)
#     This includes: cortical, subcortical_limbic, thalamus_generic, thalamic_nuclei,
#     acc_subdivision — consistent with pt2_no2 include flags.
#     Projection pathways (ATR, AR, OR) thus have proper n as their thalamic
#     endpoints pass the include flag after canonical-name category mapping.
#   - Fisher-Z FC is supplied by pt2_no02 as mean(arctanh(r)) at the voxel-pair
#     level.  The no02 input must contain mean_FC_z.
#   - Inf values in mean_dist_mm excluded row-wise
#   - Violin/scatter show pooled pair x subject data for visualization only;
#     statistical annotations reflect within-subject tests
#   - System bar y-axis = mean within-subject diff ± SEM
#   - FC ~ RD: exponential decay (a * exp(-b * RD)) fit overall and per system
#
# AUTHOR: BrainWorld pipeline
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

from cohort_settings import SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, select_subjects, select_files, filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR    = '/path/to/your/project/derivatives/results/part2/no02_wm_fc'
TEMPLATE_DIR = '/path/to/your/project/derivatives/template/AAL3'
OUTPUT_DIR   = '/path/to/your/project/derivatives/results/part2/no03_summary_stats'
# Optional descriptive RD-FC screen; not a primary test or myelin-specific measure.
RUN_EXPLORATORY_RD_FC = False
QC_EXCLUDE = {'lh.cst', 'rh.cst', 'mcp'}
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# TRACULA reference: Maffei et al. (2021), NeuroImage 245:118706
# Classification: CC and AC are commissural; MCP and bilateral FX are other.
# Projection pathways: thalamo-cortical radiations (ATR, AR, OR)
#   - endpoints span cortical AND thalamic regions (both include=1)
#   - thalamic regions are retained in this analysis (include=1 filter)
# Association pathways: ipsilateral long-range cortico-cortical connections
PATHWAY_TYPE = {
    'commissural pathways': [
        'acomm',                                          # anterior commissure
        'cc.genu', 'cc.rostrum', 'cc.splenium',          # corpus callosum
        'cc.bodyc', 'cc.bodyp', 'cc.bodypf',
        'cc.bodypm', 'cc.bodyt',
    ],
    'other pathways': ['mcp', 'lh.fx', 'rh.fx'],
    'projection pathways': [
        'lh.atr',  'rh.atr',                             # anterior thalamic radiation
        'lh.ar',   'rh.ar',                              # acoustic radiation
        'lh.or',   'rh.or',                              # optic radiation
    ],
    'association pathways': [
        'lh.af',   'rh.af',                              # arcuate fasciculus
        'lh.slf1', 'rh.slf1',                            # superior longitudinal fasciculus I
        'lh.slf2', 'rh.slf2',                            # superior longitudinal fasciculus II
        'lh.slf3', 'rh.slf3',                            # superior longitudinal fasciculus III
        'lh.uf',   'rh.uf',                              # uncinate fasciculus
        'lh.ilf',  'rh.ilf',                             # inferior longitudinal fasciculus
        'lh.mlf',  'rh.mlf',                             # middle longitudinal fasciculus
        'lh.emc',  'rh.emc',                             # extreme capsule
        'lh.fat',  'rh.fat',                             # frontal aslant tract
        'lh.cbd',  'rh.cbd',                             # cingulum bundle (dorsal)
        'lh.cbv',  'rh.cbv',                             # cingulum bundle (ventral)
    ],
}

def assign_pathway_type(tract):
    if pd.isna(tract):
        return np.nan
    for sys_name, tracts in PATHWAY_TYPE.items():
        if tract in tracts:
            return sys_name
    return 'other'

def sig_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return 'ns'

def pstr(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'

def benjamini_hochberg(pvals):
    """Return BH-FDR adjusted p values in original order."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(len(pvals), np.nan)
    valid = np.isfinite(pvals)
    if not valid.any():
        return out
    p = pvals[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[valid] = restored
    return out

# --------------------------------------------------------------------------
# 1. Load and pool all subject CSVs
# --------------------------------------------------------------------------
region_df    = pd.read_csv(os.path.join(TEMPLATE_DIR, 'aal3_region_category.csv'))
included_ids = set(region_df[region_df['include'] == 1]['roi_id'])

files = sorted(glob.glob(os.path.join(INPUT_DIR, '*_wm_present_absent_pairs.csv')))
files = select_files(files)
if not files:
    raise RuntimeError(f'No pt2_no02 subject files found in {INPUT_DIR}')
print(f"Found {len(files)} subject files")

dfs = []
for f in files:
    df = filter_subject_rows(pd.read_csv(f))
    if df['tract_name'].isin(QC_EXCLUDE).any():
        raise RuntimeError('Excluded tracts in no02 input; regenerate no02 with matching QC_EXCLUDE settings.')
    required = {'mean_FC', 'mean_FC_z', 'mean_dist_mm'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f'{f} is missing required pt2_no02 columns {sorted(missing)}. '
            'Rerun pt2_no02 before pt2_no03; do not apply Fisher-Z after averaging r.'
        )
    df = df[df['roi_id_a'].isin(included_ids) & df['roi_id_b'].isin(included_ids)]
    dfs.append(df)

tract_pool = pd.concat(dfs, ignore_index=True)
tract_pool['pair_id']      = tract_pool['roi_id_a'].astype(str) + '_' + tract_pool['roi_id_b'].astype(str)
tract_pool['pathway_type'] = tract_pool['tract_name'].apply(assign_pathway_type)
tract_pool                 = tract_pool[np.isfinite(tract_pool['mean_FC_z']) & np.isfinite(tract_pool['mean_dist_mm'])]
tract_pool                 = tract_pool.dropna(subset=['mean_FC_z', 'mean_dist_mm'])

# pt2_no02 writes one WM-present row per tract.  That is needed
# for tract/RD analyses, but the Aim 1 primary test is a REGION-PAIR analysis.
# Collapse the duplicate tract rows before estimating the WM-present effect.
pair_agg = {
    'roi_id_a': 'first', 'roi_id_b': 'first', 'roi_name_a': 'first',
    'roi_name_b': 'first', 'category_a': 'first', 'category_b': 'first',
    'mean_FC': 'first', 'mean_dist_mm': 'first', 'n_voxel_pairs': 'first',
    'mean_FC_z': 'first', 'tract_name': 'count',
}
pair_df = (tract_pool.groupby(['subject', 'pair_id', 'wm_group'], as_index=False)
           .agg(pair_agg)
           .rename(columns={'tract_name': 'n_supporting_tracts'}))
pair_df['wm_group_bin'] = (pair_df['wm_group'] == 'wm_present').astype(int)
pair_df.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3_pair_level.csv'), index=False)
tract_pool.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3_pooled_tract_level.csv'), index=False)
print(f"Subjects  : {pair_df['subject'].nunique()}")
print(f"WM-present pairs: {(pair_df['wm_group']=='wm_present').sum()}")
print(f"WM-absent pairs : {(pair_df['wm_group']=='wm_absent').sum()}")

# --------------------------------------------------------------------------
# 2. Aim 1 primary test: distance-adjusted, within-subject WM effect
# --------------------------------------------------------------------------
def one_sample_summary(values, label):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {'test': label, 'n_subjects': len(values)}
    t, p = stats.ttest_1samp(values, 0)
    se = values.std(ddof=1) / np.sqrt(len(values))
    ci_half = stats.t.ppf(.975, len(values) - 1) * se
    return {'test': label, 'n_subjects': len(values), 'mean': values.mean(),
            'se': se, 'ci95_lo': values.mean() - ci_half,
            'ci95_hi': values.mean() + ci_half, 't': t,
            'df': len(values) - 1, 'p': p}

subj_stats = []
subject_beta = {}  # Subject-specific fit reused by the pathway tests.
for subj, grp in pair_df.groupby('subject'):
    grp = grp[np.isfinite(grp['mean_FC_z']) & np.isfinite(grp['mean_dist_mm']) &
              (grp['mean_dist_mm'] > 0)].copy()
    present = grp['wm_group'].eq('wm_present')
    if present.sum() == 0 or (~present).sum() == 0 or len(grp) < 3:
        continue
    # Subject-specific log-distance residualization keeps participant as the
    # inferential unit and allows the FC-distance slope to vary by subject.
    X = np.column_stack([np.ones(len(grp)), np.log(grp['mean_dist_mm'].to_numpy())])
    beta, *_ = np.linalg.lstsq(X, grp['mean_FC_z'].to_numpy(), rcond=None)
    residual = grp['mean_FC_z'].to_numpy() - X @ beta
    subject_beta[subj] = (float(beta[0]), float(beta[1]))
    raw_diff = grp.loc[present, 'mean_FC_z'].mean() - grp.loc[~present, 'mean_FC_z'].mean()
    adjusted_diff = residual[present.to_numpy()].mean() - residual[(~present).to_numpy()].mean()
    subj_stats.append({'subject': subj, 'n_pairs': len(grp),
                       'n_wm_present_pairs': int(present.sum()),
                       'n_wm_absent_pairs': int((~present).sum()),
                       'raw_diff': raw_diff, 'distance_adjusted_diff': adjusted_diff,
                       'log_distance_slope': beta[1]})

subj_df = pd.DataFrame(subj_stats)
if subj_df.empty:
    raise RuntimeError('No subjects had both valid WM-present and WM-absent pairs.')
subj_df.to_csv(os.path.join(OUTPUT_DIR, 'pt2_no3_subject_effects.csv'), index=False)
raw_result = one_sample_summary(subj_df['raw_diff'], 'Raw within-subject WM-present minus WM-absent FC')
primary_result = one_sample_summary(subj_df['distance_adjusted_diff'],
                                    'Primary: log-distance-adjusted within-subject WM-present minus WM-absent FC')
n_subj = primary_result['n_subjects']
diffs = subj_df['distance_adjusted_diff'].to_numpy()
t_main, p_main, se_main = primary_result['t'], primary_result['p'], primary_result['se']
pd.DataFrame([primary_result, raw_result]).to_csv(
    os.path.join(OUTPUT_DIR, 'pt2_no3_distance_adjusted_stats.csv'), index=False)

print(f"\n=== Primary distance-adjusted within-subject test ===")
print(f"N subjects     : {n_subj}")
print(f"Mean difference: {primary_result['mean']:.4f} +/- {se_main:.4f} SE")
print(f"t({n_subj-1}) = {t_main:.3f}, p = {p_main:.4e}")

with open(os.path.join(OUTPUT_DIR, 'pt2_no3_within_subj_stats.txt'), 'w') as f:
    f.write('Primary analysis: subject-specific log(geodesic distance)-adjusted WM effect\n')
    f.write('One row per subject-AAL pair; multi-tract WM-present pairs are deduplicated.\n\n')
    f.write(pd.DataFrame([primary_result, raw_result]).to_string(index=False))

# --------------------------------------------------------------------------
# 3. Within-subject test per tract system
# --------------------------------------------------------------------------
SYSTEM_ORDER = ['commissural pathways', 'projection pathways', 'association pathways', 'other pathways']
SYSTEM_LABEL = {
    'commissural pathways': 'Commissural',
    'projection pathways': 'Projection',
    'association pathways': 'Association',
    'other pathways': 'Other (fornix)',
}
# Distance-adjusted, consistent with the Aim 1 primary test:
# each row is residualised with that subject's own FC ~ log(distance) fit from the
# primary analysis (estimated on deduplicated region pairs).  Within a system, a
# region pair supported by several tracts of that system is counted once.  The
# raw (unadjusted) difference is kept as a descriptive column only.
sys_pool = tract_pool[(tract_pool['mean_dist_mm'] > 0) &
                      tract_pool['subject'].isin(subject_beta.keys())].copy()
b0 = sys_pool['subject'].map(lambda s_: subject_beta[s_][0])
b1 = sys_pool['subject'].map(lambda s_: subject_beta[s_][1])
sys_pool['residual_fc_z'] = sys_pool['mean_FC_z'] - (b0 + b1 * np.log(sys_pool['mean_dist_mm']))
absent_pairs = (sys_pool[sys_pool['wm_group'] == 'wm_absent']
                .drop_duplicates(['subject', 'pair_id']))
absent_mean = absent_pairs.groupby('subject')[['residual_fc_z', 'mean_FC_z']].mean()
present_pairs = (sys_pool[sys_pool['wm_group'] == 'wm_present']
                 .drop_duplicates(['subject', 'pathway_type', 'pair_id']))

sys_results = []
for sys_name in SYSTEM_ORDER:
    pres = (present_pairs[present_pairs['pathway_type'] == sys_name]
            .groupby('subject')[['residual_fc_z', 'mean_FC_z']].mean())
    joined = pres.join(absent_mean, lsuffix='_sys', rsuffix='_abs', how='inner').dropna()
    sys_diffs = (joined['residual_fc_z_sys'] - joined['residual_fc_z_abs']).to_numpy()
    raw_diffs = (joined['mean_FC_z_sys'] - joined['mean_FC_z_abs']).to_numpy()

    if len(sys_diffs) < 5:
        print(f"{sys_name}: insufficient data (n={len(sys_diffs)}), skip")
        continue

    t, p      = stats.ttest_1samp(sys_diffs, 0)
    se        = sys_diffs.std(ddof=1) / np.sqrt(len(sys_diffs))
    sys_results.append({
        'pathway_type': sys_name,
        'mean_diff' : sys_diffs.mean(),
        'se'        : se,
        'ci_lo'     : sys_diffs.mean() - 1.96 * se,
        'ci_hi'     : sys_diffs.mean() + 1.96 * se,
        't'         : t,
        'df'        : len(sys_diffs) - 1,
        'pval'      : p,
        'n_subjects': len(sys_diffs),
        'raw_mean_diff_descriptive': raw_diffs.mean(),
    })
    print(f"{sys_name}: n_subj={len(sys_diffs)}, distance-adjusted mean_diff={sys_diffs.mean():.4f}, "
          f"t({len(sys_diffs)-1})={t:.3f}, p={p:.4e} (raw {raw_diffs.mean():.4f})")

sys_df = pd.DataFrame(sys_results, columns=[
    "pathway_type", "mean_diff", "se", "ci_lo", "ci_hi", "t", "df",
    "pval", "n_subjects", "raw_mean_diff_descriptive",
])
if not sys_df.empty:
    sys_df['p_fdr_bh'] = benjamini_hochberg(sys_df['pval'])
with open(os.path.join(OUTPUT_DIR, 'pt2_no3_system_stats.txt'), 'w') as f:
    f.write("Within-subject t-test: WM-present (by pathway type) vs WM-absent FC,\n")
    f.write("log(geodesic distance)-adjusted with each subject's primary-analysis fit.\n")
    f.write("raw_mean_diff_descriptive = unadjusted difference (descriptive only).\n")
    f.write("Tract classification: MCP and bilateral FX are other\n")
    f.write("All four pathway classes (commissural, projection, association, other = bilateral fornix)\n")
    f.write("are tested with BH-FDR correction across the four tests.\n\n")
    f.write(sys_df.to_string(index=False))

# --------------------------------------------------------------------------
# 4. Optional descriptive within-subject RD-FC screen
# --------------------------------------------------------------------------
if RUN_EXPLORATORY_RD_FC:
    rd_output_dir = os.path.join(OUTPUT_DIR, 'exploratory_rd_fc')
    os.makedirs(rd_output_dir, exist_ok=True)

    subj_rd_corr = []
    for subj, grp in tract_pool.groupby('subject'):
        wmp = grp[grp['wm_group'].eq('wm_present')].dropna(
            subset=['mean_RD', 'mean_FC_z']
        ).copy()
        if len(wmp) < 3:
            continue

        r, p = stats.spearmanr(wmp['mean_RD'], wmp['mean_FC_z'])
        if not np.isfinite(r):
            continue

        pathway_counts = wmp['pathway_type'].value_counts()
        n_unique_pairs = int(wmp['pair_id'].nunique())
        subj_rd_corr.append({
            'subject': subj,
            'spearman_r': r,
            'spearman_p': p,
            'n_tract_rows': len(wmp),
            'n_unique_region_pairs': n_unique_pairs,
            'n_reused_fc_rows': int(len(wmp) - n_unique_pairs),
            'n_commissural_rows': int(pathway_counts.get('commissural pathways', 0)),
            'n_projection_rows': int(pathway_counts.get('projection pathways', 0)),
            'n_association_rows': int(pathway_counts.get('association pathways', 0)),
        })

    rd_corr_df = pd.DataFrame(subj_rd_corr)
    if len(rd_corr_df) > 0:
        rd_corr_df.to_csv(
            os.path.join(rd_output_dir, 'rd_fc_subject_correlations.csv'),
            index=False,
        )

        r_vals = rd_corr_df['spearman_r'].to_numpy()
        t_rd, p_rd = stats.ttest_1samp(r_vals, 0)
        se_rd = r_vals.std(ddof=1) / np.sqrt(len(r_vals))
        print("\n=== Exploratory descriptive RD-FC screen ===")
        print(f"N subjects: {len(r_vals)}")
        print(f"Mean r    : {r_vals.mean():.4f} +/- {se_rd:.4f}")
        print(f"t({len(r_vals)-1}) = {t_rd:.3f}, p = {p_rd:.4e}")

        with open(os.path.join(rd_output_dir, 'rd_fc_summary.txt'), 'w') as f:
            f.write("Exploratory descriptive RD-FC screen\n")
            f.write("Per-subject Spearman r across WM-present tract rows.\n")
            f.write("Regional FC can be repeated when several tracts support the same pair.\n")
            f.write("The screen does not adjust for distance or pathway identity.\n")
            f.write("RD is not a myelin-specific measure.\n\n")
            f.write(f"N subjects : {len(r_vals)}\n")
            f.write(f"Mean r     : {r_vals.mean():.4f}\n")
            f.write(f"SE         : {se_rd:.4f}\n")
            f.write(f"t({len(r_vals)-1}) = {t_rd:.3f}\n")
            f.write(f"p          : {p_rd:.4e}\n")
    else:
        print("Exploratory RD-FC screen produced no eligible participant summaries.")
else:
    print("\nExploratory RD-FC screen skipped (RUN_EXPLORATORY_RD_FC=False).")

# --------------------------------------------------------------------------
# 5. Figures
# --------------------------------------------------------------------------

# --- Fig 1: Violin ---
fig, ax = plt.subplots(figsize=(6, 5.5))
groups  = ['wm_absent', 'wm_present']
labels  = ['WM-absent', 'WM-present']
colors  = ['#C0392B', '#1A5276']
data    = [pair_df[pair_df['wm_group'] == g]['mean_FC_z'].values for g in groups]

parts = ax.violinplot(data, positions=[0, 1], showmedians=False, showextrema=False)
for pc, col in zip(parts['bodies'], colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.18)
    pc.set_edgecolor(col)
    pc.set_linewidth(1.2)

for i, (d, col, lbl) in enumerate(zip(data, colors, labels)):
    p5, p25, p50, p75, p95 = np.percentile(d, [5, 25, 50, 75, 95])
    p1 = np.percentile(d, 1)
    ax.bar(i, p75 - p25, bottom=p25, width=0.12, color=col, alpha=0.75, zorder=3)
    ax.plot([i - 0.07, i + 0.07], [p50, p50], color='white', linewidth=2.0, zorder=4)
    ax.plot([i, i], [p5, p25], color=col, linewidth=1.2, zorder=3)
    ax.plot([i, i], [p75, p95], color=col, linewidth=1.2, zorder=3)
    ax.plot([i - 0.04, i + 0.04], [p5, p5],   color=col, linewidth=1.2, zorder=3)
    ax.plot([i - 0.04, i + 0.04], [p95, p95], color=col, linewidth=1.2, zorder=3)
    ax.plot([i - 0.18, i + 0.18], [p1, p1],
            color=col, linewidth=1.5, linestyle='--', zorder=4,
            label=f'{lbl} 1st pct={p1:.3f}')

ax.axhline(0, color='#333', linewidth=0.8, linestyle='-', zorder=1, alpha=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'WM-absent (n={len(data[0]):,})',
                    f'WM-present (n={len(data[1]):,})'], fontsize=10)
ax.set_ylabel('Mean FC (Fisher Z)', fontsize=11)
ax.set_title('FC distribution by WM group (pair-level, descriptive)', fontsize=11)
ax.legend(fontsize=8, loc='upper left', framealpha=0.8)
ax.text(0.98, 0.97,
        f'Distance-adjusted within-subj t({n_subj-1})={t_main:.2f}, {pstr(p_main)}',
        ha='right', va='top', transform=ax.transAxes, fontsize=8.5, color='#222',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pt2_no3_fig_violin.png'), dpi=300)
plt.close()
print("Saved: pt2_no3_fig_violin.png")

# --- Fig 2: FC ~ distance (log fit) ---
fig, ax = plt.subplots(figsize=(7, 5))
plot_cfg = {
    'wm_absent' : ('#D62728', 'WM-absent'),
    'wm_present': ('#1F77B4', 'WM-present'),
}
for grp_name, (col, lbl) in plot_cfg.items():
    sub = pair_df[pair_df['wm_group'] == grp_name].copy()
    sub = sub[sub['mean_dist_mm'] > 0]
    sub_plot = sub.sample(min(3000, len(sub)), random_state=42)
    ax.scatter(sub_plot['mean_dist_mm'], sub_plot['mean_FC_z'],
               alpha=0.10, s=4, color=col)
    sl, ic, r_val, _, _ = stats.linregress(np.log(sub['mean_dist_mm']), sub['mean_FC_z'])
    x_r  = np.linspace(sub['mean_dist_mm'].min(), sub['mean_dist_mm'].max(), 200)
    ax.plot(x_r, ic + sl * np.log(x_r), color=col, linewidth=2.5,
            label=f'{lbl}  log-fit: beta={sl:.4f}, r={r_val:.3f}')
ax.set_xlabel('Mean geodesic distance (mm)', fontsize=11)
ax.set_ylabel('Mean FC (Fisher Z)', fontsize=11)
ax.set_title('FC ~ geodesic distance by WM group (log fit)', fontsize=11)
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'pt2_no3_fig_scatter.png'), dpi=300)
plt.close()
print("Saved: pt2_no3_fig_scatter.png")

# --- Optional descriptive FC ~ RD plot ---
if RUN_EXPLORATORY_RD_FC and 'rd_corr_df' in locals() and len(rd_corr_df) > 0:
    pool_wmp = tract_pool[tract_pool['wm_group'].eq('wm_present')].dropna(
        subset=['mean_RD', 'mean_FC_z']
    ).copy()
    pool_wmp = pool_wmp[pool_wmp['mean_RD'] > 0]
    valid_pathways = ['commissural pathways', 'projection pathways', 'association pathways']
    pool_wmp = pool_wmp[pool_wmp['pathway_type'].isin(valid_pathways)]

    fig, ax = plt.subplots(figsize=(7, 5))

    sys_colors = {
        'commissural pathways': '#C0392B',
        'projection pathways' : '#2980B9',
        'association pathways': '#27AE60',
    }

    for sys_name, grp in pool_wmp.groupby('pathway_type'):
        col = sys_colors.get(sys_name, '#AAAAAA')
        grp_plot = grp.sample(min(4000, len(grp)), random_state=42)
        ax.scatter(grp_plot['mean_RD'], grp_plot['mean_FC_z'],
                   alpha=0.20, s=6, color=col,
                   label=f'{sys_name} (rows={len(grp):,})')

    ax.set_xlabel('Mean RD (mm2/s)', fontsize=11)
    ax.set_ylabel('Mean FC (Fisher Z)', fontsize=11)
    ax.set_title('Descriptive FC and RD across WM-present tract rows', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    ax.text(0.05, 0.97,
            f'Mean participant Spearman r={r_vals.mean():.3f}; descriptive screen',
            ha='left', va='top', transform=ax.transAxes, fontsize=8.5, color='#222',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(rd_output_dir, 'rd_fc_descriptive.png'), dpi=300)
    plt.close()
    print("Saved: exploratory_rd_fc/rd_fc_descriptive.png")

# --- Fig 3: Within-subject diff by pathway type ---
# Plot all four pathway classes against the same participant-level WM-absent pool.
sys_df_plot = (sys_df.set_index('pathway_type')
               .reindex([s_ for s_ in SYSTEM_ORDER if s_ in set(sys_df['pathway_type'])])
               .reset_index())
if len(sys_df_plot) > 0:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(sys_df_plot))
    ax.bar(x, sys_df_plot['mean_diff'], color='#1A5276', alpha=0.80, width=0.55)
    ax.errorbar(x, sys_df_plot['mean_diff'], yerr=sys_df_plot['se'],
                fmt='none', color='#222', capsize=4, linewidth=1.5)
    ax.axhline(0, color='#888', linewidth=1, linestyle='--')

    lo = min(0.0, (sys_df_plot['mean_diff'] - sys_df_plot['se']).min())
    hi = max(0.0, (sys_df_plot['mean_diff'] + sys_df_plot['se']).max())
    pad = 0.18 * (hi - lo if hi > lo else 1e-3)
    ax.set_ylim(lo - pad, hi + pad)

    for idx, (_, row) in enumerate(sys_df_plot.iterrows()):
        # place stars above bar if positive, below bar if negative
        if row['mean_diff'] >= 0:
            star_y = row['mean_diff'] + row['se'] + 0.15 * pad
            va_pos = 'bottom'
        else:
            star_y = row['mean_diff'] - row['se'] - 0.15 * pad
            va_pos = 'top'
        ax.text(idx, star_y, sig_stars(row['p_fdr_bh']),
                ha='center', va=va_pos, fontsize=12, color='#222', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{SYSTEM_LABEL.get(r['pathway_type'], r['pathway_type'])}\nn={int(r['n_subjects'])}"
         for _, r in sys_df_plot.iterrows()], fontsize=10)
    ax.set_ylabel('Distance-adjusted FC difference\n(WM-present minus WM-absent, Fisher Z)', fontsize=10)
    ax.set_title('Within-subject distance-adjusted FC difference by pathway type', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'pt2_no3_fig_system.png'), dpi=300)
    plt.close()
    print("Saved: pt2_no3_fig_system.png")


# --------------------------------------------------------------------------
# 6. Generate pathway classification text file
# --------------------------------------------------------------------------
pathway_txt_path = os.path.join(OUTPUT_DIR, 'tracula_pathway_classification.txt')



with open(pathway_txt_path, 'w') as f:
    f.write("TRACULA White Matter Pathway Classification\n")
    f.write("=" * 43 + "\n\n")
    f.write("Reference:\n")
    f.write("  Maffei et al. (2021). NeuroImage 245:118706.\n")
    f.write("  https://doi.org/10.1016/j.neuroimage.2021.118706\n\n")
    f.write("Pipeline classification: MCP and bilateral FX are OTHER.\n")
    f.write("MCP and bilateral CST are excluded from group analysis.\n\n")
    f.write("Region filter (consistent with pt2_no2):\n")
    f.write("  include==1 from aal3_region_category.csv\n")
    f.write("  (cortical, subcortical_limbic, thalamus_generic,\n")
    f.write("   thalamic_nuclei, acc_subdivision)\n\n")
    f.write("System tests: all four pathway classes; other comprises bilateral fornix.\n")
    f.write("See pt2_no3_system_stats.txt for distance-adjusted contrasts.\n\n")

    counts = dict.fromkeys(PATHWAY_TYPE, 0)
    qc_counts = dict.fromkeys(PATHWAY_TYPE, 0)

    for ptype, tracts in PATHWAY_TYPE.items():
        f.write("=" * 43 + "\n")
        f.write(f"{ptype.upper()}\n")
        f.write("=" * 43 + "\n")
        for t in tracts:
            qc_note = "  [excluded by QC_EXCLUDE]" if t in QC_EXCLUDE else ""
            f.write(f"  {t:<18s}{qc_note}\n")
            counts[ptype] += 1
            if t in QC_EXCLUDE:
                qc_counts[ptype] += 1
        f.write("\n")

    f.write("=" * 43 + "\n")
    f.write("OBSERVED TRACT COUNTS\n")
    observed = sorted(tract_pool.loc[tract_pool['wm_group'] == 'wm_present', 'tract_name'].dropna().unique())
    f.write(f"  Distinct tracts with finite FC and distance: {len(observed)}\n")
    f.write("  " + ", ".join(observed) + "\n")
    f.write("  Acquisition: 40 tracts (42 minus bilateral CST). Analysis: 39 (minus MCP); eligibility may reduce observed counts.\n")
    total_working = sum(counts.values())
    total_qc = sum(qc_counts.values())
    for ptype, tracts in PATHWAY_TYPE.items():
        n_analysis = sum(t in tracts for t in observed)
        f.write(f"    {ptype:<28s}: {n_analysis}\n")

print(f"Saved: {pathway_txt_path}")

print("\nDone.")
