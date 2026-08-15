#!/usr/bin/env python3
# =============================================================================
# pt2_no3_summary_stats_v8.py
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
#     System test: same approach per pathway type (commissural/association/projection)
#     RD test    : per-subject Spearman r (FC ~ RD within WM-present pairs)
#                  tested against zero via one-sample t-test
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
#   - pt2_no3_rd_stats.txt           : RD-FC Spearman + t-test
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
#     endpoints are included (thalamus_generic=81/82, thalamic_nuclei=121-150).
#   - Fisher Z transform applied to all FC values (arctanh, clipped +-0.9999)
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

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR    = '/path/to/your/project/derivatives/pt2_group_results/no2_wm_fc'
TEMPLATE_DIR = '/path/to/your/project/derivatives/template/AAL3'
OUTPUT_DIR   = '/path/to/your/project/derivatives/pt2_group_results/no3_summary_stats_v8'
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pathway type classification per Maffei et al. (2021), NeuroImage 245:118706
# "The 42 fibers are divided into three types:
#  commissural pathways, projection pathways, and association pathways."
# Commissural pathways include: CC, AC, MCP, FX (Maffei et al., 2021)
# Projection pathways: thalamo-cortical radiations (ATR, AR, OR)
#   - endpoints span cortical AND thalamic regions (both include=1)
#   - thalamic regions are retained in this analysis (include=1 filter)
# Association pathways: ipsilateral long-range cortico-cortical connections
PATHWAY_TYPE = {
    'commissural pathways': [
        'acomm',                                          # anterior commissure
        'mcp',                                            # middle cerebellar peduncle
        'lh.fx',   'rh.fx',                              # fornix
        'cc.genu', 'cc.rostrum', 'cc.splenium',          # corpus callosum
        'cc.bodyc', 'cc.bodyp', 'cc.bodypf',
        'cc.bodypm', 'cc.bodyt',
    ],
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
if not files:
    raise RuntimeError(f'No pt2_no02 subject files found in {INPUT_DIR}')
print(f"Found {len(files)} subject files")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df = df[df['roi_id_a'].isin(included_ids) & df['roi_id_b'].isin(included_ids)]
    dfs.append(df)

tract_pool = pd.concat(dfs, ignore_index=True)
tract_pool['pair_id']      = tract_pool['roi_id_a'].astype(str) + '_' + tract_pool['roi_id_b'].astype(str)
tract_pool['pathway_type'] = tract_pool['tract_name'].apply(assign_pathway_type)
tract_pool                 = tract_pool[np.isfinite(tract_pool['mean_FC']) & np.isfinite(tract_pool['mean_dist_mm'])]
tract_pool                 = tract_pool.dropna(subset=['mean_FC', 'mean_dist_mm'])
tract_pool['mean_FC_z']    = np.arctanh(np.clip(tract_pool['mean_FC'], -0.9999, 0.9999))

# pt2_no02 deliberately writes one WM-present row per tract.  That is needed
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
sys_results = []
for sys_name in ['commissural pathways', 'projection pathways', 'association pathways']:
    sys_diffs = []
    for subj, grp in tract_pool.groupby('subject'):
        mu_sys = grp[(grp['wm_group'] == 'wm_present') &
                     (grp['pathway_type'] == sys_name)]['mean_FC_z'].mean()
        mu_abs = grp[grp['wm_group'] == 'wm_absent']['mean_FC_z'].mean()
        if np.isfinite(mu_sys) and np.isfinite(mu_abs):
            sys_diffs.append(mu_sys - mu_abs)

    if len(sys_diffs) < 5:
        print(f"{sys_name}: insufficient data (n={len(sys_diffs)}), skip")
        continue

    sys_diffs = np.array(sys_diffs)
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
    })
    print(f"{sys_name}: n_subj={len(sys_diffs)}, mean_diff={sys_diffs.mean():.4f}, "
          f"t({len(sys_diffs)-1})={t:.3f}, p={p:.4e}")

sys_df = pd.DataFrame(sys_results)
if not sys_df.empty:
    sys_df['p_fdr_bh'] = benjamini_hochberg(sys_df['pval'])
with open(os.path.join(OUTPUT_DIR, 'pt2_no3_system_stats.txt'), 'w') as f:
    f.write("Within-subject t-test: WM-present (by pathway type) vs WM-absent FC\n")
    f.write("Pathway classification: Maffei et al., 2021, NeuroImage 245:118706\n")
    f.write("NOTE: Commissural and association pathways shown in main figure.\n")
    f.write("      Projection pathways retained here for supplementary material.\n\n")
    f.write(sys_df.to_string(index=False))

# --------------------------------------------------------------------------
# 4. Within-subject RD-FC Spearman correlation
# --------------------------------------------------------------------------
subj_rd_corr = []
for subj, grp in tract_pool.groupby('subject'):
    wmp = grp[(grp['wm_group'] == 'wm_present')].dropna(subset=['mean_RD', 'mean_FC_z'])
    if len(wmp) < 3:
        continue
    r, p = stats.spearmanr(wmp['mean_RD'], wmp['mean_FC_z'])
    if np.isfinite(r):
        subj_rd_corr.append({'subject': subj, 'spearman_r': r, 'p': p})

rd_corr_df = pd.DataFrame(subj_rd_corr)
t_rd, p_rd, r_vals = np.nan, np.nan, np.array([])
if len(rd_corr_df) > 0:
    r_vals       = rd_corr_df['spearman_r'].values
    t_rd, p_rd   = stats.ttest_1samp(r_vals, 0)
    se_rd        = r_vals.std(ddof=1) / np.sqrt(len(r_vals))
    print(f"\n=== Within-subject RD-FC Spearman r ===")
    print(f"N subjects: {len(r_vals)}")
    print(f"Mean r    : {r_vals.mean():.4f} +/- {se_rd:.4f}")
    print(f"t({len(r_vals)-1}) = {t_rd:.3f}, p = {p_rd:.4e}")
    with open(os.path.join(OUTPUT_DIR, 'pt2_no3_rd_stats.txt'), 'w') as f:
        f.write("Within-subject Spearman r (FC ~ RD, WM-present pairs)\n")
        f.write(f"N subjects : {len(r_vals)}\n")
        f.write(f"Mean r     : {r_vals.mean():.4f}\n")
        f.write(f"SE         : {se_rd:.4f}\n")
        f.write(f"t({len(r_vals)-1}) = {t_rd:.3f}\n")
        f.write(f"p          : {p_rd:.4e}\n")

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

# --- Fig 2b: FC ~ RD (exponential decay + per-system fits) ---
pool_wmp = tract_pool[tract_pool['wm_group'] == 'wm_present'].dropna(subset=['mean_RD', 'mean_FC_z'])
pool_wmp = pool_wmp[pool_wmp['mean_RD'] > 0].copy()

# 데이터 필터링: 'other'에 해당하는 경로(lh.cst, rh.cst 등) 명시적 제외
valid_pathways = ['commissural pathways', 'projection pathways', 'association pathways']
pool_wmp = pool_wmp[pool_wmp['pathway_type'].isin(valid_pathways)].copy()

if len(pool_wmp) > 50:
    fig, ax = plt.subplots(figsize=(7, 5))

    sys_colors = {
        'commissural pathways': '#C0392B',
        'projection pathways' : '#2980B9',
        'association pathways': '#27AE60',
    }

    for sys_name, grp in pool_wmp.groupby('pathway_type'):
        col = sys_colors.get(sys_name, '#AAAAAA')
        ax.scatter(grp['mean_RD'], grp['mean_FC_z'],
                   alpha=0.22, s=6, color=col, label=f'{sys_name} (n={len(grp)})')

    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    # overall fit (이제 'other' 경로가 완전히 배제된 데이터로만 피팅을 진행합니다)
    try:
        popt, _ = curve_fit(exp_decay, pool_wmp['mean_RD'], pool_wmp['mean_FC_z'],
                            p0=[0.15, 2000], maxfev=10000,
                            bounds=([0, 0], [np.inf, np.inf]))
        x_fit = np.linspace(pool_wmp['mean_RD'].min(), pool_wmp['mean_RD'].max(), 300)
        ax.plot(x_fit, exp_decay(x_fit, *popt), color='black', linewidth=2.5,
                label=f'Overall exp fit: a={popt[0]:.3f}, b={popt[1]:.0f}')
    except Exception as e:
        print(f"Overall exp fit failed: {e}")

    # per-system fits (dashed)
    for sys_name in valid_pathways:
        grp = pool_wmp[pool_wmp['pathway_type'] == sys_name]
        if len(grp) < 10:
            continue
        col = sys_colors.get(sys_name, '#AAAAAA')
        try:
            popt_s, _ = curve_fit(exp_decay, grp['mean_RD'], grp['mean_FC_z'],
                                  p0=[0.15, 2000], maxfev=10000,
                                  bounds=([0, 0], [np.inf, np.inf]))
            x_s = np.linspace(grp['mean_RD'].min(), grp['mean_RD'].max(), 200)
            ax.plot(x_s, exp_decay(x_s, *popt_s),
                    color=col, linewidth=1.8, linestyle='--')
        except Exception as e:
            print(f"{sys_name} exp fit failed: {e}")

    ax.set_xlabel('Mean RD (mm2/s)', fontsize=11)
    ax.set_ylabel('Mean FC (Fisher Z)', fontsize=11)
    ax.set_title('FC ~ RD within WM-present pairs (higher RD = lower myelination)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')

    if np.isfinite(t_rd) and len(r_vals) > 0:
        ax.text(0.05, 0.97,
                f'Within-subj Spearman r={r_vals.mean():.3f}, '
                f't({len(r_vals)-1})={t_rd:.2f}, {pstr(p_rd)}',
                ha='left', va='top', transform=ax.transAxes, fontsize=8.5, color='#222',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'pt2_no3_fig_fc_rd.png'), dpi=300)
    plt.close()
    print("Saved: pt2_no3_fig_fc_rd.png")

# --- Fig 3: Within-subject diff by pathway type ---
# Main figure: commissural + association pathways only (both cortical-cortical).
# Projection pathways are computed in stats but excluded from main figure
# (thalamo-cortical endpoints make direct comparison with cortical-cortical
# WM-absent pool non-equivalent; see pt2_no3_system_stats.txt for full results).
sys_df_plot = sys_df[sys_df['pathway_type'].isin(
    ['commissural pathways', 'association pathways'])].reset_index(drop=True)
if len(sys_df_plot) > 0:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(sys_df_plot))
    ax.bar(x, sys_df_plot['mean_diff'], color='#1A5276', alpha=0.80, width=0.5)
    ax.errorbar(x, sys_df_plot['mean_diff'], yerr=sys_df_plot['se'],
                fmt='none', color='#222', capsize=4, linewidth=1.5)
    ax.axhline(0, color='#888', linewidth=1, linestyle='--')

    ymax = (sys_df_plot['mean_diff'] + sys_df_plot['se']).max()
    ax.set_ylim(bottom=-0.0015, top=ymax + 0.003)

    for idx, (_, row) in enumerate(sys_df_plot.iterrows()):
        # place stars above bar if positive, below bar if negative
        if row['mean_diff'] >= 0:
            star_y = row['mean_diff'] + row['se'] + 0.0003
            va_pos = 'bottom'
        else:
            star_y = row['mean_diff'] - row['se'] - 0.0003
            va_pos = 'top'
        ax.text(idx, star_y, sig_stars(row['p_fdr_bh']),
                ha='center', va=va_pos, fontsize=12, color='#222', fontweight='bold')
        ax.text(idx, -0.001,
                f"n={int(row['n_subjects'])} subj  df={int(row['df'])}",
                ha='center', va='top', fontsize=7.5, color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(sys_df_plot['pathway_type'], fontsize=11)
    ax.set_ylabel('Mean FC difference\n(WM-present minus WM-absent, Fisher Z)', fontsize=10)
    ax.set_title('Within-subject FC difference by pathway type', fontsize=11)
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

QC_EXCLUDE_TRACTS = {'acomm', 'rh.atr'}

with open(pathway_txt_path, 'w') as f:
    f.write("TRACULA White Matter Pathway Classification\n")
    f.write("=" * 43 + "\n\n")
    f.write("Reference:\n")
    f.write("  Maffei et al. (2021). NeuroImage 245:118706.\n")
    f.write("  https://doi.org/10.1016/j.neuroimage.2021.118706\n\n")
    f.write("Classification:\n")
    f.write('  "The 42 fibers are divided into three types:\n')
    f.write('   commissural pathways, projection pathways, and association pathways."\n')
    f.write('  "The commissural pathways include the anterior commissure (AC),\n')
    f.write('   middle cerebellar peduncle (MCP), fornix (FX) and corpus callosum."\n\n')
    f.write("Region filter (consistent with pt2_no2):\n")
    f.write("  include==1 from aal3_region_category.csv\n")
    f.write("  (cortical, subcortical_limbic, thalamus_generic,\n")
    f.write("   thalamic_nuclei, acc_subdivision)\n\n")
    f.write("Main figure: commissural + association pathways only\n")
    f.write("Supplementary: projection pathways (see pt2_no3_system_stats.txt)\n\n")

    counts = {'commissural pathways': 0, 'projection pathways': 0, 'association pathways': 0}
    qc_counts = {'commissural pathways': 0, 'projection pathways': 0, 'association pathways': 0}

    for ptype, tracts in PATHWAY_TYPE.items():
        f.write("=" * 43 + "\n")
        f.write(f"{ptype.upper()}\n")
        f.write("=" * 43 + "\n")
        for t in tracts:
            qc_note = "  [QC excluded]" if t in QC_EXCLUDE_TRACTS else ""
            f.write(f"  {t:<18s}{qc_note}\n")
            counts[ptype] += 1
            if t in QC_EXCLUDE_TRACTS:
                qc_counts[ptype] += 1
        f.write("\n")

    f.write("=" * 43 + "\n")
    f.write("EXCLUDED FROM ACQUISITION\n")
    f.write("=" * 43 + "\n")
    f.write("  lh.cst  Left Corticospinal Tract   [removed]\n")
    f.write("  rh.cst  Right Corticospinal Tract  [removed]\n\n")

    f.write("=" * 43 + "\n")
    f.write("TRACT COUNT SUMMARY\n")
    f.write("=" * 43 + "\n")
    total_working = sum(counts.values())
    total_qc      = sum(qc_counts.values())
    f.write(f"  Total in TRACULA atlas       : 42\n")
    f.write(f"  Removed (lh/rh.cst)         :  2\n")
    f.write(f"  Working set                  : {total_working}\n")
    f.write(f"  QC excluded                  :  {total_qc}\n")
    f.write(f"  Analysis set                 : {total_working - total_qc}\n")
    for ptype, tracts in PATHWAY_TYPE.items():
        n_analysis = counts[ptype] - qc_counts[ptype]
        f.write(f"    {ptype:<28s}: {n_analysis}\n")

print(f"Saved: {pathway_txt_path}")

print("\nDone.")
