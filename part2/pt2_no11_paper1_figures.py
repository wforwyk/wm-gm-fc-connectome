#!/usr/bin/env python3
"""Generate publication figures from the frozen Paper 1 Part 2 outputs.

This script intentionally reads only group-level outputs from pt2_no03 and
pt2_no10.  It never re-runs a statistical test, so the figures remain an
auditable rendering of the saved primary analyses.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# USER SETTINGS
# =============================================================================
NO3_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no3_summary_stats_v8'
NO10_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no10_matched_endpoint_stats'
OUTPUT_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no11_paper1_figures'
RANDOM_SEED = 20260814
# =============================================================================

BLUE = '#1A5276'
RED = '#C0392B'
GOLD = '#C78B00'
GRAY = '#666666'

def require(path):
    if not os.path.exists(path):
        raise RuntimeError(f'Required input not found: {path}')
    return path

def p_text(p):
    if not np.isfinite(p):
        return 'p unavailable'
    return 'p<0.001' if p < .001 else f'p={p:.3f}'

def mean_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return mean, mean - 1.96 * se, mean + 1.96 * se

def save(fig, filename, caption):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=350, bbox_inches='tight')
    plt.close(fig)
    return {'file': filename, 'caption': caption}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    manifest = []

    # Figure 1 — subject-level, distance-adjusted Aim 1 effect.
    subj = pd.read_csv(require(os.path.join(NO3_DIR, 'pt2_no3_subject_effects.csv')))
    stats = pd.read_csv(require(os.path.join(NO3_DIR, 'pt2_no3_distance_adjusted_stats.csv')))
    adjusted = subj['distance_adjusted_diff'].dropna().to_numpy()
    primary = stats.loc[stats.test.str.startswith('Primary:')].iloc[0]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.violinplot(adjusted, positions=[0], showmedians=False, showextrema=False)
    for body in ax.collections:
        body.set_facecolor(BLUE); body.set_edgecolor(BLUE); body.set_alpha(.20)
    jitter = rng.uniform(-.09, .09, len(adjusted))
    ax.scatter(jitter, adjusted, s=13, color=BLUE, alpha=.38, linewidths=0)
    mean, lo, hi = mean_ci(adjusted)
    ax.errorbar(0, mean, yerr=[[mean-lo], [hi-mean]], fmt='o', color='black', capsize=5, zorder=5)
    ax.axhline(0, color=GRAY, linestyle='--', linewidth=1)
    ax.set_xticks([0]); ax.set_xticklabels(['WM-present − WM-absent'])
    ax.set_ylabel('Distance-adjusted FC difference (Fisher Z)')
    ax.set_title('Aim 1: WM-linked pairs retain higher FC beyond distance')
    ax.text(.97, .96, f"n={int(primary.n_subjects)} subjects\n"
            f"mean={primary['mean']:.4f} [{primary.ci95_lo:.4f}, {primary.ci95_hi:.4f}]\n"
            f"t({int(primary.df)})={primary.t:.2f}, {p_text(primary.p)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#CCCCCC'))
    ax.spines[['top', 'right']].set_visible(False)
    manifest.append(save(fig, 'fig2_aim1_distance_adjusted_effect.png',
                         'Subject-level distance-adjusted WM-present FC effect.'))

    # Figure 2 — matched endpoint interaction, with subject as the displayed unit.
    interaction = pd.read_csv(require(os.path.join(NO10_DIR, 'aim2_subject_interaction.csv')))
    if 'subject' not in interaction.columns:
        interaction = interaction.rename(columns={interaction.columns[0]: 'subject'})
    required = {'endpoint', 'matched_non_endpoint', 'endpoint_specific_distant_preference'}
    if not required.issubset(interaction.columns):
        raise RuntimeError(f'Aim 2 interaction CSV lacks: {required-set(interaction.columns)}')
    primary2 = pd.read_csv(require(os.path.join(NO10_DIR, 'aim2_primary_stats.csv'))).iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    pair = interaction[['endpoint', 'matched_non_endpoint']].dropna().to_numpy()
    for a, b in pair:
        ax.plot([0, 1], [b, a], color='#BFC9CA', linewidth=.65, alpha=.6, zorder=1)
    ax.scatter(np.zeros(len(pair)), pair[:, 1], s=16, color=GRAY, alpha=.58, label='Matched non-endpoint')
    ax.scatter(np.ones(len(pair)), pair[:, 0], s=16, color=GOLD, alpha=.68, label='Endpoint')
    for x, values, color in [(0, pair[:,1], GRAY), (1, pair[:,0], GOLD)]:
        m, lo, hi = mean_ci(values)
        ax.errorbar(x, m, yerr=[[m-lo], [hi-m]], fmt='o', color='black', capsize=5, zorder=4)
    ax.axhline(0, color=GRAY, linestyle='--', linewidth=1)
    ax.set_xlim(-.35, 1.35); ax.set_xticks([0, 1]); ax.set_xticklabels(['Matched control', 'Endpoint'])
    ax.set_ylabel('Distant − adjacent FC (Fisher Z)')
    ax.set_title('Aim 2: endpoint-specific preference for WM-connected targets')
    ax.text(.97, .96, f"n={int(primary2.n_subjects)} subjects\n"
            f"interaction={primary2['mean']:.4f} [{primary2.ci95_lo:.4f}, {primary2.ci95_hi:.4f}]\n"
            f"t({int(primary2.df)})={primary2.t:.2f}, {p_text(primary2.p)}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#CCCCCC'))
    ax.legend(frameon=False, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    manifest.append(save(fig, 'fig3_aim2_matched_endpoint_interaction.png',
                         'Subject-level matched endpoint by target-type interaction.'))

    # Figure 3 — matching balance (Love plot).
    balance = pd.read_csv(require(os.path.join(NO10_DIR, 'aim2_matching_balance_all.csv')))
    if 'eligible_primary' not in balance.columns:
        raise RuntimeError('Balance output lacks eligible_primary; rerun pt2_no10.')
    balance = balance[balance['eligible_primary']].copy()
    if balance.empty:
        raise RuntimeError('No primary-eligible endpoint units available for balance plot.')
    cols = ['smd_wm_distance', 'smd_gm_probability']
    melt = balance[cols].melt(var_name='covariate', value_name='smd').dropna()
    labels = {'smd_wm_distance': 'WM-boundary distance', 'smd_gm_probability': 'GM probability'}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for y, col in enumerate(cols):
        vals = melt.loc[melt.covariate.eq(col), 'smd'].to_numpy()
        ax.scatter(vals, np.full(len(vals), y) + rng.uniform(-.10, .10, len(vals)),
                   color=BLUE, alpha=.35, s=14)
        ax.scatter(np.median(vals), y, color='black', marker='D', s=35, zorder=4)
    ax.axvline(-.10, color=RED, linestyle='--', linewidth=1)
    ax.axvline(.10, color=RED, linestyle='--', linewidth=1, label='|SMD| = 0.10 criterion')
    ax.axvline(0, color=GRAY, linewidth=.8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([labels[x] for x in cols])
    ax.set_xlabel('Standardized mean difference: endpoint − matched control')
    ax.set_title('Aim 2 matching balance among primary-eligible endpoint units')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    manifest.append(save(fig, 'fig4_aim2_matching_balance.png',
                         'Covariate balance of primary-eligible matched endpoint units.'))

    # Figure 4 — Aim 2 effect-size summary for contrasts on the FC scale.
    endpoint = pd.read_csv(require(os.path.join(NO10_DIR, 'aim2_endpoint_control_stats.csv')))
    effect_rows = [{'label': 'Endpoint × target interaction', 'mean': primary2['mean'],
                    'lo': primary2['ci95_lo'], 'hi': primary2['ci95_hi'], 'p': primary2['p']}]
    for _, row in endpoint.iterrows():
        if np.isfinite(row.get('mean', np.nan)):
            effect_rows.append({'label': f"Endpoint − control: {row['target_type']}",
                                'mean': row['mean'], 'lo': row['ci95_lo'], 'hi': row['ci95_hi'],
                                'p': row.get('p_fdr_bh', row.get('p', np.nan))})
    effects = pd.DataFrame(effect_rows).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, max(3.3, 1.0 + .65 * len(effects))))
    y = np.arange(len(effects))
    ax.errorbar(effects['mean'], y,
                xerr=[effects['mean']-effects['lo'], effects['hi']-effects['mean']],
                fmt='o', color=BLUE, capsize=4)
    ax.axvline(0, color=GRAY, linestyle='--', linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(effects['label'])
    ax.set_xlabel('FC effect (Fisher Z; mean and 95% CI)')
    ax.set_title('Aim 2 endpoint effects after matching')
    for yi, row in effects.iterrows():
        ax.text(row['hi'], yi, '  ' + p_text(row['p']), va='center', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    manifest.append(save(fig, 'fig5_aim2_effect_summary.png',
                         'Matched Aim 2 FC effects; target-wise p values are BH-FDR adjusted.'))

    pd.DataFrame(manifest).to_csv(os.path.join(OUTPUT_DIR, 'paper1_figure_manifest.csv'), index=False)
    print(f'Wrote {len(manifest)} Paper 1 figures to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
