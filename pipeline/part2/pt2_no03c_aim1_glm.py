#!/usr/bin/env python3
# =============================================================================
# pt2_no03c_aim1_glm.py
#
# AIM 1 / PURPOSE:
#   Split the FC of each region pair into a GM-route part (geodesic distance) and a
#   WM-route part (a reconstructed tract between the two regions) with a
#   participant-level regression.
#
# MODEL:
#   Model, fitted separately in each participant (OLS); one region pair = one row:
#       FC_z = b0 + S(log d) + b2 * WM + error                       (primary)
#       FC_z = b0 + S(log d) + b2 * WM + b3 * WM * (log d - ref)     (exploratory)
#   FC_z   = mean Fisher-z FC over all voxel pairs of the region pair (pt2_no02)
#   S()    = restricted cubic spline, 5 knots at the 5, 27.5, 50, 72.5 and 95th
#            percentiles of log distance pooled over participants. The knots are the
#            same for every participant.
#   WM     = 1 if a reconstructed tract is assigned to the region pair, else 0
#   b2     = WM effect at the same distance (primary WM-route estimate).
#   b3     = linear change of the WM effect with log distance (exploratory).
#   GM route = the fitted distance curve S(). It is summarised as the drop of
#            predicted FC (WM-absent) from the 10th to the 90th percentile of distance.
#
# HYPOTHESES (fixed in the paper):
#   H1  FC falls with distance (GM route): gm_drop_q10_q90 > 0.
#   H2  At the same distance, region pairs with a tract have higher FC: b2 > 0.
#   Distance dependence of the WM effect is exploratory (no direction fixed).
#
# DISTANCE PROFILE (model-free check of the shape, same 12 quantiles as Figure 2):
#   Within each participant the distance curve is fitted on WM-absent pairs only.
#   For every within-participant distance quantile the WM effect is the mean residual
#   of WM-present pairs minus the mean residual of WM-absent pairs. Quantiles in which
#   a participant has no WM-present pair are missing for that participant.
#
# GROUP LEVEL:
#   The participant is the unit of inference. Each coefficient is tested against zero
#   across participants with a one-sample t test (t-based 95% CI, d_z = mean / SD), in
#   Fisher-z units. Pairs are never pooled across participants. Tests are unadjusted.
#   Primary tests: gm_drop_q10_q90 (H1) and b2_wm (H2).
#
# VARIANCE PARTITION (order independent for the main effects):
#   unique GM   = R2(distance + WM) - R2(WM only)
#   unique WM   = R2(distance + WM) - R2(distance only)
#   shared      = R2(distance only) + R2(WM only) - R2(distance + WM)
#   interaction = R2(full) - R2(distance + WM)
#   "shared" can be negative (suppression); it is reported as computed.
#   WM-present pairs are about 1% of all pairs, so R2 of WM is small by construction.
#   Compare effect sizes (b2 against the distance drop) rather than R2 alone.
#
# R-UNITS:
#   Coefficients are not converted to r. For each participant, FC is predicted for
#   WM-absent and WM-present pairs at the 10th/50th/90th percentile of the pooled
#   distance distribution, and tanh() turns the predicted Fisher z into r.
#
# SENSITIVITY (separate outputs, not primary):
#   S1 log-linear distance
#   S2 adjacency indicator added to the primary model (AAL3 regions sharing a border)
#   S3 comparison with the log-linear residual contrast (delta_s)
#   S4 spline with 4 knots instead of 5
#   Scope alternatives: change CORTICAL_CATEGORIES in USER SETTINGS.
#
# CLASS-SPECIFIC WM EFFECT (pathway class):
#   For each class (commissural, association, projection, other) the same spline model is
#   fitted on all WM-absent pairs plus the WM-present pairs of that class; WM-present pairs of
#   the other classes are left out of that fit. The coefficient is the WM effect of that class
#   at the same distance (b2_<class>). A pair supported by tracts of two classes counts in both.
#   A participant needs at least MIN_PRESENT_CLASS pairs of the class. The planned contrast
#   is b2_commissural - b2_association (participants with both).
#
# INPUT : pt2_no3_pair_level.csv written by
#         pt2_no03_stats.py.
#         pt2_no3_pooled_tract_level.csv from the same run (pathway class of each pair).
#         Optional: aal3_adjacent_pairs.csv (S2).
# OUTPUT: new folder only; no existing result is overwritten.
#
# USAGE: python pt2_no03c_aim1_glm.py
#         (check)  python pt2_no03c_aim1_glm.py --selftest
# =============================================================================
import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from scipy import stats
from cohort_settings import filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
ROOT = '/path/to/your/project/derivatives/'
INPUT_CSV = ROOT + 'results/part2/no03_summary_stats/pt2_no3_pair_level.csv'
ADJACENCY_CSV = ROOT + 'template/AAL3/aal3_adjacent_pairs.csv'
TRACT_LEVEL_CSV = ROOT + 'results/part2/no03_summary_stats/pt2_no3_pooled_tract_level.csv'
OUTPUT_DIR = ROOT + 'results/part2/no03c_aim1_glm_spline'
CORTICAL_CATEGORIES = {'cortical'}      # both regions of a pair must be in this set
MIN_PRESENT = 5                         # a participant needs this many WM-present pairs
MIN_ABSENT = 100                        # and this many WM-absent pairs
MIN_PRESENT_CLASS = 1                   # class-specific fit: pairs of that class per participant
                                        # (same as pt2_no03b, so the class results stay comparable;
                                        # a class with 1-2 pairs per participant gives noisy
                                        # participant estimates, read the group n and CI)
KNOT_PERCENTILES = (5, 27.5, 50, 72.5, 95)      # primary spline (5 knots, 4 columns)
KNOT_PERCENTILES_SENS = (5, 35, 65, 95)         # S4 (4 knots, 3 columns)
DISTANCE_QUANTILES = (0.10, 0.50, 0.90)         # of the pooled distance distribution
N_PROFILE_BINS = 12                             # within-participant quantiles, as in Figure 2
FIG_DPI = 300
# =============================================================================

REQUIRED = ['subject', 'pair_id', 'wm_group_bin', 'roi_id_a', 'roi_id_b',
            'category_a', 'category_b', 'mean_FC_z', 'mean_dist_mm']
TAGS = ('q10', 'q50', 'q90')
CLASS_SHORT = {'commissural pathways': 'commissural', 'association pathways': 'association',
               'projection pathways': 'projection', 'other pathways': 'other'}


# --------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------
def ols(X, y):
    """OLS with an intercept column already in X. Returns coefficients and R2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - float(np.sum(resid ** 2)) / sst
    return beta, r2


def rcs_basis(x, knots):
    """Restricted (natural) cubic spline basis (Harrell), len(knots) - 1 columns.
    Column 1 is x itself. The other columns are zero-curvature beyond the outer knots."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if len(k) < 3:
        raise ValueError('A restricted cubic spline needs at least 3 knots')
    t1, tk1, tk = k[0], k[-2], k[-1]

    def pos3(u):
        return np.clip(u, 0.0, None) ** 3

    cols = [x]
    for tj in k[:-2]:
        cols.append((pos3(x - tj) - pos3(x - tk1) * (tk - tj) / (tk - tk1)
                     + pos3(x - tk) * (tk1 - tj) / (tk - tk1)) / (tk - t1) ** 2)
    return np.column_stack(cols)


def summarise(values):
    """One-sample t summary of a vector across participants."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return dict(n=n, mean=np.nan, sd=np.nan, se=np.nan, t=np.nan, df=n - 1,
                    p=np.nan, ci_lo=np.nan, ci_hi=np.nan, d_z=np.nan)
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    se = sd / np.sqrt(n)
    t, p = stats.ttest_1samp(v, 0.0)
    half = stats.t.ppf(0.975, n - 1) * se
    return dict(n=n, mean=mean, sd=sd, se=se, t=float(t), df=n - 1, p=float(p),
                ci_lo=mean - half, ci_hi=mean + half, d_z=mean / sd if sd > 0 else np.nan)


def load_pairs(path, categories, max_subjects=None):
    df = pd.read_csv(path, usecols=REQUIRED)
    df = filter_subject_rows(df)
    keep = df['category_a'].isin(categories) & df['category_b'].isin(categories)
    df = df[keep & np.isfinite(df['mean_FC_z']) & np.isfinite(df['mean_dist_mm'])
            & (df['mean_dist_mm'] > 0)].copy()
    if max_subjects:
        subjects = sorted(df['subject'].unique())[:max_subjects]
        df = df[df['subject'].isin(subjects)]
    return df


def load_class_membership(path):
    """{subject: {class name: set of pair_id}} for WM-present pairs, from the pooled tract-level table."""
    if not path or not os.path.isfile(path):
        return None
    t = pd.read_csv(path, usecols=['subject', 'pair_id', 'wm_group', 'pathway_type'])
    t = filter_subject_rows(t)
    t = t[t['wm_group'] == 'wm_present'].dropna(subset=['pathway_type'])
    t = t[['subject', 'pair_id', 'pathway_type']].drop_duplicates()
    out = {}
    for (subject, cls), g in t.groupby(['subject', 'pathway_type']):
        out.setdefault(subject, {})[cls] = set(g['pair_id'])
    return out


def fit_class(g, members, ctx):
    """WM effect of one pathway class at the same distance. Rows: all WM-absent pairs plus the
    WM-present pairs of this class. Returns (coefficient, number of class pairs)."""
    w = g['wm_group_bin'].to_numpy(float)
    in_class = g['pair_id'].isin(members).to_numpy() & (w == 1)
    n_cls = int(in_class.sum())
    if n_cls < MIN_PRESENT_CLASS:
        return np.nan, n_cls
    rows = (w == 0) | in_class
    y = g['mean_FC_z'].to_numpy(float)[rows]
    logd = np.log(g['mean_dist_mm'].to_numpy(float))[rows]
    X = np.column_stack([np.ones(int(rows.sum())), rcs_basis(logd, ctx['knots']),
                         in_class[rows].astype(float)])
    beta, _ = ols(X, y)
    return float(beta[-1]), n_cls


def load_adjacency(path):
    if not path or not os.path.isfile(path):
        return None
    adj = pd.read_csv(path)
    if 'both_included' in adj.columns:
        adj = adj[adj['both_included'] == 1]
    lo = np.minimum(adj['roi_id_a'], adj['roi_id_b'])
    hi = np.maximum(adj['roi_id_a'], adj['roi_id_b'])
    return set(zip(lo.astype(int), hi.astype(int)))


# --------------------------------------------------------------------------
# One participant
# --------------------------------------------------------------------------
def fit_participant(g, adjacency, ctx):
    y = g['mean_FC_z'].to_numpy(float)
    logd = np.log(g['mean_dist_mm'].to_numpy(float))
    w = g['wm_group_bin'].to_numpy(float)
    one = np.ones_like(y)
    knots, knots_sens, ref, Xq = ctx['knots'], ctx['knots_sens'], ctx['ref'], ctx['Xq']

    Xd = rcs_basis(logd, knots)
    nd = Xd.shape[1]
    xc = logd - ref

    _, r2_gm = ols(np.column_stack([one, Xd]), y)
    _, r2_wm = ols(np.column_stack([one, w]), y)
    b_main, r2_main = ols(np.column_stack([one, Xd, w]), y)
    b_full, r2_full = ols(np.column_stack([one, Xd, w, xc * w]), y)

    row = dict(subject=g['subject'].iloc[0], n_pairs=len(g), n_present=int(w.sum()),
               n_absent=int((1 - w).sum()),
               b0=b_main[0], b2_wm=b_main[1 + nd],
               b3_wm_x_distance=b_full[-1], b2_wm_with_interaction=b_full[1 + nd],
               r2_distance_only=r2_gm, r2_wm_only=r2_wm, r2_main=r2_main, r2_full=r2_full,
               unique_gm=r2_main - r2_wm, unique_wm=r2_main - r2_gm,
               shared=r2_gm + r2_wm - r2_main, interaction_increment=r2_full - r2_main)
    for i in range(nd):
        row[f'sp{i + 1}'] = b_main[1 + i]

    # Predictions at pooled distance quantiles (primary additive model)
    z_abs_q = b_main[0] + Xq @ b_main[1:1 + nd]
    for tag, z_abs in zip(TAGS, z_abs_q):
        z_pre = z_abs + b_main[1 + nd]
        row[f'z_absent_{tag}'], row[f'z_present_{tag}'] = z_abs, z_pre
        row[f'r_absent_{tag}'], row[f'r_present_{tag}'] = np.tanh(z_abs), np.tanh(z_pre)
        row[f'r_diff_{tag}'] = np.tanh(z_pre) - np.tanh(z_abs)
    row['gm_drop_q10_q90'] = row['z_absent_q10'] - row['z_absent_q90']

    # S1: log-linear distance (participant-centred)
    c = logd - logd.mean()
    b_ll, _ = ols(np.column_stack([one, c, w, c * w]), y)
    row.update(loglin_b1_distance=b_ll[1], loglin_b2_wm=b_ll[2], loglin_b3_interaction=b_ll[3])

    # S3: log-linear residual contrast (one log-linear slope over all pairs)
    b_gm_ll, _ = ols(np.column_stack([one, c]), y)
    resid = y - np.column_stack([one, c]) @ b_gm_ll
    row['delta_s_old'] = float(resid[w == 1].mean() - resid[w == 0].mean())

    # S4: spline with 4 knots
    Xs = rcs_basis(logd, knots_sens)
    b_s4, _ = ols(np.column_stack([one, Xs, w]), y)
    row['spl4_b2_wm'] = b_s4[1 + Xs.shape[1]]

    # S2: adjacency indicator added to the primary model
    if adjacency is not None:
        a = np.array([(min(i, j), max(i, j)) in adjacency
                      for i, j in zip(g['roi_id_a'].astype(int), g['roi_id_b'].astype(int))], dtype=float)
        row['n_adjacent'] = int(a.sum())
        row['n_present_adjacent'] = int((a * w).sum())
        if 0 < a.sum() < len(a):
            b_adj, r2_adj = ols(np.column_stack([one, Xd, w, a]), y)
            row.update(adj_b2_wm=b_adj[1 + nd], adj_b_adjacent=b_adj[2 + nd], r2_with_adjacency=r2_adj)
    return row


def profile_participant(g, knots, n_bins):
    """WM effect in within-participant distance quantiles. The distance curve is fitted on
    WM-absent pairs only; the effect is the mean residual of present minus absent pairs."""
    y = g['mean_FC_z'].to_numpy(float)
    logd = np.log(g['mean_dist_mm'].to_numpy(float))
    w = g['wm_group_bin'].to_numpy(float)
    absent = w == 0
    X = np.column_stack([np.ones_like(y), rcs_basis(logd, knots)])
    beta, *_ = np.linalg.lstsq(X[absent], y[absent], rcond=None)
    resid = y - X @ beta
    ranks = pd.Series(logd).rank(method='first', pct=True).to_numpy(float)
    bins = np.minimum((ranks * n_bins).astype(int), n_bins - 1)
    subject = g['subject'].iloc[0]
    rows = []
    for b in range(n_bins):
        m = bins == b
        p, a = m & (w == 1), m & (w == 0)
        rp = float(resid[p].mean()) if p.any() else np.nan
        ra = float(resid[a].mean()) if a.any() else np.nan
        rows.append(dict(subject=subject, bin=b + 1, n_present=int(p.sum()), n_absent=int(a.sum()),
                         raw_present=float(y[p].mean()) if p.any() else np.nan,
                         raw_absent=float(y[a].mean()) if a.any() else np.nan,
                         res_present=rp, res_absent=ra,
                         contrast=rp - ra if (p.any() and a.any()) else np.nan,
                         dist_mm=float(np.exp(logd[m]).mean())))
    return rows


# --------------------------------------------------------------------------
# Whole analysis
# --------------------------------------------------------------------------
def run(df, output_dir, adjacency, make_figures=True, class_members=None):
    os.makedirs(output_dir, exist_ok=True)
    keep, skipped = [], []
    for subject, g in df.groupby('subject', sort=True):
        n_pre = int((g['wm_group_bin'] == 1).sum())
        n_abs = int((g['wm_group_bin'] == 0).sum())
        if n_pre < MIN_PRESENT or n_abs < MIN_ABSENT:
            skipped.append(dict(subject=subject, n_present=n_pre, n_absent=n_abs))
        else:
            keep.append(subject)
    df = df[df['subject'].isin(keep)]
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(output_dir, 'aim1_glm_skipped_participants.csv'), index=False)

    # Fixed knots, reference distance and prediction distances (pooled over participants)
    logd_all = np.log(df['mean_dist_mm'].to_numpy(float))
    ctx = dict(knots=np.percentile(logd_all, KNOT_PERCENTILES),
               knots_sens=np.percentile(logd_all, KNOT_PERCENTILES_SENS),
               ref=float(logd_all.mean()))
    qlogd = np.quantile(logd_all, DISTANCE_QUANTILES)
    ctx['Xq'] = rcs_basis(qlogd, ctx['knots'])

    rows, prof = [], []
    for subject, g in df.groupby('subject', sort=True):
        row = fit_participant(g, adjacency, ctx)
        if class_members is not None:
            members = class_members.get(subject, {})
            for cls, short in CLASS_SHORT.items():
                row[f'b2_{short}'], row[f'n_present_{short}'] = fit_class(g, members.get(cls, set()), ctx)
        rows.append(row)
        prof.extend(profile_participant(g, ctx['knots'], N_PROFILE_BINS))
    eff = pd.DataFrame(rows)
    eff.to_csv(os.path.join(output_dir, 'aim1_glm_participant_effects.csv'), index=False)
    prof = pd.DataFrame(prof)
    prof.to_csv(os.path.join(output_dir, 'aim1_glm_distance_profile_participant.csv'), index=False)

    tab = [dict(model='primary', term='gm_drop_q10_q90', **summarise(eff['gm_drop_q10_q90'])),
           dict(model='primary', term='b2_wm', **summarise(eff['b2_wm']))]
    for tag in TAGS:
        tab.append(dict(model='primary', term=f'r_diff_{tag}', **summarise(eff[f'r_diff_{tag}'])))
    tab.append(dict(model='exploratory', term='b3_wm_x_distance', **summarise(eff['b3_wm_x_distance'])))
    tab.append(dict(model='exploratory', term='b2_wm_with_interaction', **summarise(eff['b2_wm_with_interaction'])))
    for t in ('loglin_b1_distance', 'loglin_b2_wm', 'loglin_b3_interaction'):
        tab.append(dict(model='S1_loglinear', term=t, **summarise(eff[t])))
    if 'b2_commissural' in eff:
        for short in CLASS_SHORT.values():
            tab.append(dict(model='class_specific', term=f'b2_{short}', **summarise(eff[f'b2_{short}'])))
        tab.append(dict(model='class_specific', term='b2_commissural - b2_association',
                        **summarise(eff['b2_commissural'] - eff['b2_association'])))
    if 'adj_b2_wm' in eff:
        for t in ('adj_b2_wm', 'adj_b_adjacent'):
            tab.append(dict(model='S2_adjacency', term=t, **summarise(eff[t])))
    tab.append(dict(model='S3_old_contrast', term='delta_s_old', **summarise(eff['delta_s_old'])))
    tab.append(dict(model='S3_old_minus_main', term='delta_s_old - b2_wm',
                    **summarise(eff['delta_s_old'] - eff['b2_wm'])))
    tab.append(dict(model='S4_four_knot_spline', term='spl4_b2_wm', **summarise(eff['spl4_b2_wm'])))
    group = pd.DataFrame(tab)
    group.to_csv(os.path.join(output_dir, 'aim1_glm_group_stats.csv'), index=False)

    parts = ['r2_distance_only', 'r2_wm_only', 'r2_main', 'r2_full',
             'unique_gm', 'unique_wm', 'shared', 'interaction_increment']
    r2 = pd.DataFrame([dict(measure=m, **summarise(eff[m])) for m in parts])
    r2.to_csv(os.path.join(output_dir, 'aim1_glm_r2_partition.csv'), index=False)

    pred = []
    for tag, q in zip(TAGS, qlogd):
        pred.append(dict(quantile=tag, dist_mm=float(np.exp(q)),
                         r_absent=float(eff[f'r_absent_{tag}'].mean()),
                         r_present=float(eff[f'r_present_{tag}'].mean()),
                         **{f'diff_{k}': v for k, v in summarise(eff[f'r_diff_{tag}']).items()}))
    pd.DataFrame(pred).to_csv(os.path.join(output_dir, 'aim1_glm_predicted_r.csv'), index=False)

    # Group profile per distance quantile
    gp = []
    for b, gb in prof.groupby('bin'):
        s = summarise(gb['contrast'])
        sp_, sa_ = summarise(gb['raw_present']), summarise(gb['raw_absent'])
        gp.append(dict(bin=b, dist_mm_mean=float(gb['dist_mm'].mean()),
                       mean_n_present_per_participant=float(gb['n_present'].mean()),
                       n_participants_with_contrast=s['n'], contrast_mean=s['mean'], contrast_se=s['se'],
                       contrast_ci_lo=s['ci_lo'], contrast_ci_hi=s['ci_hi'], contrast_p=s['p'],
                       raw_present_mean=sp_['mean'], raw_present_se=sp_['se'], raw_present_n=sp_['n'],
                       raw_absent_mean=sa_['mean'], raw_absent_se=sa_['se']))
    gp = pd.DataFrame(gp)
    gp.to_csv(os.path.join(output_dir, 'aim1_glm_distance_profile_group.csv'), index=False)

    with open(os.path.join(output_dir, 'aim1_glm_summary.txt'), 'w') as f:
        f.write('Aim 1 GLM, spline version (pt2_no03c_aim1_glm)\n')
        f.write(f'Cortical categories: {sorted(CORTICAL_CATEGORIES)}\n')
        f.write(f'Participants analysed: {len(eff)}; skipped: {len(skipped)}\n')
        f.write(f'Pairs per participant (min/max): {int(eff.n_pairs.min())}/{int(eff.n_pairs.max())}\n')
        f.write(f'WM-present pairs per participant (mean/min/max): '
                f'{eff.n_present.mean():.1f}/{int(eff.n_present.min())}/{int(eff.n_present.max())}\n')
        f.write('Spline knots (mm): ' + ', '.join(f'{np.exp(k):.1f}' for k in ctx['knots']) + '\n')
        f.write(f'Reference distance for the interaction (mm): {np.exp(ctx["ref"]):.1f}\n\n')
        f.write('Group statistics (Fisher-z units unless stated):\n')
        f.write(group.to_string(index=False, float_format=lambda x: f'{x:.6g}') + '\n\n')
        f.write('R2 partition across participants:\n')
        f.write(r2.to_string(index=False, float_format=lambda x: f'{x:.6g}') + '\n\n')
        f.write('Predicted r (mean across participants; distances are pooled quantiles):\n')
        f.write(pd.DataFrame(pred).to_string(index=False, float_format=lambda x: f'{x:.6g}') + '\n\n')
        f.write('WM effect by within-participant distance quantile (distance curve fitted on WM-absent pairs):\n')
        f.write(gp.to_string(index=False, float_format=lambda x: f'{x:.6g}') + '\n')

    if make_figures:
        figures(df, eff, r2, gp, ctx, output_dir)
    return eff, group, r2


def figures(df, eff, r2, gp, ctx, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Predicted FC against distance (mm), from each participant's primary model
    lo, hi = np.percentile(df['mean_dist_mm'], [2, 98])
    grid = np.linspace(lo, hi, 120)
    Bg = rcs_basis(np.log(grid), ctx['knots'])
    sp_cols = [c for c in eff.columns if c.startswith('sp') and c[2:].isdigit()]
    pred_abs, pred_pre = [], []
    for _, r in eff.iterrows():
        z_abs = r['b0'] + Bg @ r[sp_cols].to_numpy(float)
        pred_abs.append(np.tanh(z_abs))
        pred_pre.append(np.tanh(z_abs + r['b2_wm']))
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, arr, colour in (('WM-absent', np.array(pred_abs), '#555555'),
                               ('WM-present', np.array(pred_pre), '#c0392b')):
        m = arr.mean(0)
        half = 1.96 * arr.std(0, ddof=1) / np.sqrt(len(arr))
        ax.plot(grid, m, color=colour, label=label)
        ax.fill_between(grid, m - half, m + half, color=colour, alpha=0.2)
    ax.set_xlabel('Geodesic distance (mm)')
    ax.set_ylabel('Predicted FC (r)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'aim1_glm_fc_by_distance.png'), dpi=FIG_DPI)
    plt.close(fig)

    # Variance partition
    order = ['unique_gm', 'unique_wm', 'shared', 'interaction_increment']
    labels = ['GM distance\n(unique)', 'WM presence\n(unique)', 'Shared', 'Distance x WM']
    sub = r2.set_index('measure').loc[order]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(range(4), sub['mean'], yerr=[sub['mean'] - sub['ci_lo'], sub['ci_hi'] - sub['mean']],
           color=['#2e75b6', '#c0392b', '#999999', '#7f6000'], capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Explained variance (R2)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'aim1_glm_r2_partition.png'), dpi=FIG_DPI)
    plt.close(fig)

    # Distance profile: raw FC and WM effect per within-participant distance quantile
    x = gp['bin'].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for lab, col, c in (('WM-absent', 'raw_absent', '#555555'), ('WM-present', 'raw_present', '#c0392b')):
        m, se = gp[f'{col}_mean'].to_numpy(float), gp[f'{col}_se'].to_numpy(float)
        ax.plot(x, m, marker='o', markersize=3.5, color=c, label=lab)
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.2, linewidth=0)
    ax.set_xlabel('Within-participant distance quantile')
    ax.set_ylabel('Mean FC (Fisher z)')
    ax.legend(frameon=False)
    ax = axes[1]
    ax.axhline(0, color='#6E6E6E', linewidth=0.9, linestyle='--')
    ax.plot(x, gp['contrast_mean'], marker='o', markersize=3.5, color='#6A0DAD')
    ax.fill_between(x, gp['contrast_ci_lo'], gp['contrast_ci_hi'], color='#6A0DAD', alpha=0.2, linewidth=0)
    ax.set_xlabel('Within-participant distance quantile')
    ax.set_ylabel('WM-present minus WM-absent (Fisher z)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'aim1_glm_distance_profile.png'), dpi=FIG_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Self-test on synthetic data with known coefficients (not a pipeline run)
# --------------------------------------------------------------------------
def selftest():
    # Spline basis: right number of columns, linear beyond the last knot, exact for a line
    knots = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xs = np.linspace(5.5, 8.0, 6)
    B = rcs_basis(xs, knots)
    assert B.shape == (6, 4)
    assert np.allclose(np.diff(B, n=2, axis=0), 0, atol=1e-9), 'spline columns must be linear beyond the last knot'
    xline = np.linspace(0.5, 6.0, 40)
    Bl = np.column_stack([np.ones_like(xline), rcs_basis(xline, knots)])
    coef, *_ = np.linalg.lstsq(Bl, 2.0 - 0.5 * xline, rcond=None)
    assert np.allclose(Bl @ coef, 2.0 - 0.5 * xline, atol=1e-9)

    # Data: a non-linear distance curve (steep, then flat), a constant WM effect, no interaction
    rng = np.random.default_rng(0)
    b2_class = {'commissural pathways': 0.020, 'association pathways': 0.005}
    p_comm = 0.3
    b2_true = p_comm * b2_class['commissural pathways'] + (1 - p_comm) * b2_class['association pathways']
    members = {}
    rows = []
    ids = np.arange(1, 61)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]][:600]
    for s in range(1, 41):
        for (a, b) in pairs:
            d = float(np.exp(rng.normal(np.log(90), 0.30)))
            w = int(rng.random() < 0.05)
            cls = 'commissural pathways' if rng.random() < p_comm else 'association pathways'
            if w:
                members.setdefault(s, {}).setdefault(cls, set()).add(f'{a}_{b}')
            fc = 0.002 + 0.06 * np.exp(-d / 25.0) + (b2_class[cls] if w else 0.0) + rng.normal(0, 0.03)
            rows.append(dict(subject=s, pair_id=f'{a}_{b}', wm_group_bin=w, roi_id_a=a, roi_id_b=b,
                             category_a='cortical', category_b='cortical', mean_dist_mm=d, mean_FC_z=fc))
    df = pd.DataFrame(rows)
    adjacency = {(1, 2), (3, 4), (5, 6), (10, 11)}
    out = tempfile.mkdtemp(prefix='aim1_glm_selftest_')
    eff, group, r2 = run(df, out, adjacency, make_figures=True, class_members=members)
    g = group.set_index(['model', 'term'])['mean']
    assert abs(g[('class_specific', 'b2_commissural')] - 0.020) < 0.006, g[('class_specific', 'b2_commissural')]
    assert abs(g[('class_specific', 'b2_association')] - 0.005) < 0.004, g[('class_specific', 'b2_association')]
    assert abs(g[('class_specific', 'b2_commissural - b2_association')] - 0.015) < 0.007
    n_proj = group.set_index(['model', 'term']).loc[('class_specific', 'b2_projection'), 'n']
    assert n_proj == 0, 'a class without any pairs must give no estimate'
    # Participant-level estimates come from about 30 WM-present pairs, so allow a wide tolerance.
    assert abs(g[('primary', 'b2_wm')] - b2_true) < 0.004, f"b2 {g[('primary', 'b2_wm')]:.4f}"
    assert g[('primary', 'gm_drop_q10_q90')] > 0
    assert abs(g[('exploratory', 'b3_wm_x_distance')]) < 0.02
    assert np.allclose(eff.unique_gm + eff.unique_wm + eff.shared, eff.r2_main, atol=1e-10)
    assert np.allclose(eff.r2_main + eff.interaction_increment, eff.r2_full, atol=1e-10)
    assert (eff.n_pairs == len(pairs)).all()
    for name in ('aim1_glm_fc_by_distance.png', 'aim1_glm_distance_profile.png',
                 'aim1_glm_distance_profile_group.csv', 'aim1_glm_summary.txt'):
        assert os.path.isfile(os.path.join(out, name)), name
    gp = pd.read_csv(os.path.join(out, 'aim1_glm_distance_profile_group.csv'))
    assert len(gp) == N_PROFILE_BINS
    print(f'Self-test passed (synthetic data). b2 estimate {g[("primary", "b2_wm")]:.4f} '
          f'(truth {b2_true}). Output: {out}')


def main():
    ap = argparse.ArgumentParser(description='Aim 1 spline GLM')
    ap.add_argument('--selftest', action='store_true', help='run the synthetic-data check and exit')
    ap.add_argument('--input', default=INPUT_CSV)
    ap.add_argument('--output', default=OUTPUT_DIR)
    ap.add_argument('--adjacency', default=ADJACENCY_CSV)
    ap.add_argument('--tract-level', default=TRACT_LEVEL_CSV, help='pooled tract-level table (pathway class)')
    ap.add_argument('--cortical-categories', nargs='+', default=sorted(CORTICAL_CATEGORIES))
    ap.add_argument('--max-subjects', type=int, default=None, help='testing only')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    df = load_pairs(args.input, set(args.cortical_categories), args.max_subjects)
    adjacency = load_adjacency(args.adjacency)
    if adjacency is None:
        print('Adjacency table not found: sensitivity S2 is skipped.')
    class_members = load_class_membership(args.tract_level)
    if class_members is None:
        print('Tract-level table not found: class-specific effects are skipped.')
    eff, group, _ = run(df, args.output, adjacency, class_members=class_members)
    print(f'Participants analysed: {len(eff)}')
    print(group[group.model == 'primary'][['term', 'n', 'mean', 'ci_lo', 'ci_hi', 'p']].to_string(index=False))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
