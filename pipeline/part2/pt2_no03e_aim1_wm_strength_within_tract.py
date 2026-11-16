#!/usr/bin/env python3
# =============================================================================
# pt2_no03e_aim1_wm_strength_within_tract.py
#
# OPTIONAL EXPLORATORY ANALYSIS:
#   Relate participant differences in tract FA, MD, AD and RD to the WM effect
#   within the same tract. Run only when this additional analysis is wanted;
#   it is not a required pipeline stage.
#
# DEFINITIONS:
#   1. GM route removed. S(log d) is a restricted cubic spline fitted on the WM-ABSENT cortico-cortical
#      pairs of the participant (same knots and basis as pt2_no03c). For every WM-present pair
#      e_pair = FC_z - S(log d).
#   2. Participant x tract table. e_tract = mean of e_pair over the pairs the tract is assigned to; the
#      tract's FA, MD, AD, RD come from the participant's own pathstats file (Part 1).
#   3. Models (one metric at a time; the four metrics are strongly collinear):
#        M1  e_it = b * x_it + tract fixed effect                       (within-tract across participants)
#        M2  e_it = b * x_it + tract fixed effect + participant fixed effect
#      M2 removes everything that shifts all tracts of one participant together (overall FC level,
#      signal quality, motion, site, age) and keeps only how far a tract deviates from its own usual
#      value relative to the participant's other tracts. M2 is the stricter control.
#      Standard errors are cluster-robust by participant (CR1), t reference with (participants - 1) df.
#      slope = change in the WM effect (Fisher-z) per 1 SD of the within-tract deviation of the metric
#      (SD of x after removing the tract mean; the same SD is used for M1 and M2).
#   4. Sets of tracts: pooled commissural + association (main set for this optional analysis), commissural and association
#      slopes from one interaction model and their difference, and all tracts (descriptive).
#      Projection and other (fornix) enter only the
#      all-tracts set.
#   5. Descriptive checks: per-tract correlation across participants, the number
#      of tracts whose correlation has the same sign (two-sided binomial test, tract is the unit), and
#      leave-one-tract-out slopes (is one tract driving the pooled result?).
#
# MULTIPLE TESTING: Benjamini-Hochberg q within each family: {model} x {term} over the four metrics.
#
# WHAT THIS CANNOT SHOW:
#   Within-tract differences of a tract metric between participants partly reflect measurement noise
#   (template-based reconstruction, partial volume) and individual anatomy. A significant slope is an
#   association between a participant's tract metric and the same participant's WM effect; it does not
#   show that microstructure causes the effect.
#
# INPUT : pt2_no3_pair_level.csv and pt2_no3_pooled_tract_level.csv written by
#         pt2_no03_stats.py; {BATCH_DIR}/{subj}/wm/derived/pathstats/{subj}_pathstats.csv
# OUTPUT: new folder only (no existing result is overwritten):
#         aim1_wmstrength_within_participant_tract.csv   participant x tract table (effect, metrics)
#         aim1_wmstrength_within_group_stats.csv         slope tests (M1, M2) with cluster-robust SE, BH q
#         aim1_wmstrength_within_tract_corr.csv          per-tract correlation across participants
#         aim1_wmstrength_within_tract_sign_test.csv     sign test across tracts
#         aim1_wmstrength_within_leave_one_tract_out.csv influence of single tracts
#         aim1_wmstrength_within_summary.txt
#         fig_aim1_wm_strength_within.{png,pdf,svg}      figure (Arial; SVG text stays editable)
#
# USAGE: python pt2_no03e_aim1_wm_strength_within_tract.py
#         (check)  python pt2_no03e_aim1_wm_strength_within_tract.py --selftest
#         (redraw) python pt2_no03e_aim1_wm_strength_within_tract.py --figures-only
# Needs pt2_no03c_aim1_glm.py in the same folder (spline basis, t summary).
# =============================================================================
import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from cohort_settings import filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
ROOT = '/path/to/your/project/derivatives/'
INPUT_CSV = ROOT + 'results/part2/no03_summary_stats/pt2_no3_pair_level.csv'
TRACT_LEVEL_CSV = ROOT + 'results/part2/no03_summary_stats/pt2_no3_pooled_tract_level.csv'
BATCH_DIR = ROOT + 'batch'
OUTPUT_DIR = ROOT + 'results/part2/no03e_aim1_wm_strength_within_tract'
TRACT_SUFFIX = '_avg16_syn_bbr'
CORTICAL_CATEGORIES = {'cortical'}      # both regions of a pair must be in this set (as no03c)
METRICS = {'FA': 'FA_Avg', 'MD': 'MD_Avg', 'AD': 'AD_Avg', 'RD': 'RD_Avg'}
MIN_PRESENT = 5                         # as no03c: WM-present pairs per participant
MIN_ABSENT = 100                        # as no03c: WM-absent pairs per participant
MIN_ROWS_TRACT = 30                     # participants needed for a per-tract correlation
KNOT_PERCENTILES = (5, 27.5, 50, 72.5, 95)      # as no03c (5 knots, 4 columns)
CLASS_SHORT = {'commissural pathways': 'commissural', 'association pathways': 'association',
               'projection pathways': 'projection', 'other pathways': 'other'}
CLASS_COLOUR = {'commissural': '#1E5B9C', 'association': '#1F8A6A', 'projection': '#C4612A', 'other': '#A0579F'}
FIG_DPI = 400
# =============================================================================

REQUIRED_PAIR = ['subject', 'pair_id', 'wm_group_bin', 'category_a', 'category_b', 'mean_FC_z', 'mean_dist_mm']
REQUIRED_TRACT = ['subject', 'pair_id', 'wm_group', 'category_a', 'category_b', 'tract_name', 'pathway_type']


def _import_no03c():
    path = Path(__file__).resolve().parent / 'pt2_no03c_aim1_glm.py'
    spec = importlib.util.spec_from_file_location('no03c', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NO03C = _import_no03c()
rcs_basis = NO03C.rcs_basis


def bh(p):
    """Benjamini-Hochberg adjusted p values (NaN stays NaN)."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    adj = p[ok][order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    q[ok] = out
    return q


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_pairs(path, categories, max_subjects=None):
    df = pd.read_csv(path, usecols=REQUIRED_PAIR, dtype={'subject': str})
    df = filter_subject_rows(df)
    keep = df['category_a'].isin(categories) & df['category_b'].isin(categories)
    df = df[keep & np.isfinite(df['mean_FC_z']) & np.isfinite(df['mean_dist_mm']) & (df['mean_dist_mm'] > 0)].copy()
    if max_subjects:
        subjects = sorted(df['subject'].unique())[:max_subjects]
        df = df[df['subject'].isin(subjects)]
    return df


def load_tracts(path, categories):
    t = pd.read_csv(path, usecols=REQUIRED_TRACT, dtype={'subject': str})
    t = filter_subject_rows(t)
    t = t[(t['wm_group'] == 'wm_present') & t['category_a'].isin(categories) & t['category_b'].isin(categories)]
    t = t.dropna(subset=['tract_name', 'pathway_type'])
    t = t[t['tract_name'].astype(str) != '']
    t['cls'] = t['pathway_type'].map(CLASS_SHORT)
    return t[['subject', 'pair_id', 'tract_name', 'cls']].drop_duplicates()


def read_pathstats(subject, batch_dir):
    path = os.path.join(batch_dir, subject, 'wm', 'derived', 'pathstats', f'{subject}_pathstats.csv')
    if not os.path.isfile(path):
        return None
    ps = pd.read_csv(path)
    ps['tract_clean'] = ps['tract'].astype(str).str.replace(TRACT_SUFFIX, '', regex=False)
    if len([c for c in METRICS.values() if c in ps.columns]) < len(METRICS):
        return None
    return ps.set_index('tract_clean')[list(METRICS.values())]


def pair_effects(g, knots):
    """WM effect of every WM-present pair: FC_z minus the spline fitted on WM-absent pairs."""
    y = g['mean_FC_z'].to_numpy(float)
    logd = np.log(g['mean_dist_mm'].to_numpy(float))
    w = g['wm_group_bin'].to_numpy(float)
    X = np.column_stack([np.ones_like(y), rcs_basis(logd, knots)])
    beta, *_ = np.linalg.lstsq(X[w == 0], y[w == 0], rcond=None)
    resid = y - X @ beta
    return pd.Series(resid[w == 1], index=g['pair_id'].to_numpy()[w == 1])


def build_long_table(df, tracts, batch_dir):
    """One row per participant x tract: WM effect and the participant's own tract metrics."""
    keep, skipped = [], []
    for subject, g in df.groupby('subject', sort=True):
        n_pre, n_abs = int((g['wm_group_bin'] == 1).sum()), int((g['wm_group_bin'] == 0).sum())
        if n_pre < MIN_PRESENT or n_abs < MIN_ABSENT:
            skipped.append(dict(subject=subject, reason='too few pairs', n_present=n_pre, n_absent=n_abs))
        else:
            keep.append(subject)
    df = df[df['subject'].isin(keep)]
    knots = np.percentile(np.log(df['mean_dist_mm'].to_numpy(float)), KNOT_PERCENTILES)
    t_by_subject = {s: g for s, g in tracts.groupby('subject')}
    rows = []
    for subject, g in df.groupby('subject', sort=True):
        ps = read_pathstats(subject, batch_dir)
        tg = t_by_subject.get(subject)
        if ps is None:
            skipped.append(dict(subject=subject, reason='pathstats file missing or without FA/MD/AD/RD'))
            continue
        if tg is None:
            skipped.append(dict(subject=subject, reason='no WM-present tract rows'))
            continue
        eff = pair_effects(g, knots)
        tg = tg[tg['pair_id'].isin(eff.index)].copy()
        tg['effect'] = eff.reindex(tg['pair_id']).to_numpy()
        per = (tg.groupby('tract_name').agg(effect=('effect', 'mean'), n_pairs=('pair_id', 'nunique'),
                                            cls=('cls', 'first')).reset_index().rename(columns={'tract_name': 'tract'}))
        per = per[per['tract'].isin(ps.index)]
        for metric, col in METRICS.items():
            per[metric] = ps.loc[per['tract'], col].to_numpy(float)
        per = per.dropna(subset=['effect', *METRICS])
        per.insert(0, 'subject', subject)
        rows.append(per)
    if not rows:
        raise SystemExit('No participant could be analysed (check the pathstats folder and the tract-level table).')
    return pd.concat(rows, ignore_index=True), skipped, knots


# --------------------------------------------------------------------------
# Fixed-effects regression with participant-clustered standard errors
# --------------------------------------------------------------------------
def _demean(a, codes, n):
    sums = np.bincount(codes, weights=a, minlength=n)
    cnt = np.bincount(codes, minlength=n)
    return a - (sums / np.maximum(cnt, 1))[codes]


def _residualise(v, tcode, ntr, scode, nsub, twoway):
    v = np.asarray(v, float).copy()
    if not twoway:
        return _demean(v, tcode, ntr)
    for _ in range(1000):                       # alternating projections (unbalanced panel)
        old = v
        v = _demean(_demean(v, tcode, ntr), scode, nsub)
        if np.max(np.abs(v - old)) < 1e-13:
            break
    return v


def fe_ols(y, X, tcode, scode, twoway):
    """y ~ X + tract FE (+ participant FE), CR1 cluster-robust by participant.
    Returns beta, vcov, df (clusters - 1), n_clusters."""
    ntr, nsub = int(tcode.max()) + 1, int(scode.max()) + 1
    yt = _residualise(y, tcode, ntr, scode, nsub, twoway)
    Xt = np.column_stack([_residualise(X[:, j], tcode, ntr, scode, nsub, twoway) for j in range(X.shape[1])])
    p = Xt.shape[1]
    xtx = Xt.T @ Xt
    if np.linalg.matrix_rank(xtx) < p:
        return np.full(p, np.nan), np.full((p, p), np.nan), 0, 0
    bread = np.linalg.inv(xtx)
    beta = bread @ (Xt.T @ yt)
    e = yt - Xt @ beta
    S = np.column_stack([np.bincount(scode, weights=Xt[:, j] * e, minlength=nsub) for j in range(p)])
    G = int((np.bincount(scode, minlength=nsub) > 0).sum())
    n = len(y)
    k = p + len(np.unique(tcode)) - 1 + ((G - 1) if twoway else 0)
    corr = G / (G - 1) * (n - 1) / max(n - k, 1)
    V = bread @ (S.T @ S) @ bread * corr
    return beta, V, G - 1, G


def _codes(d):
    return (pd.factorize(d['tract'])[0].astype(int), pd.factorize(d['subject'])[0].astype(int))


def _row(model, set_name, metric, term, b, se, df_, unit, sd_y, n_rows, n_part, n_tr):
    t = b / se if se and np.isfinite(se) and se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df_)) if np.isfinite(t) and df_ > 0 else np.nan
    half = stats.t.ppf(0.975, df_) * se if df_ > 0 else np.nan
    return dict(model=model, set=set_name, metric=metric, term=term, n_rows=n_rows, n_participants=n_part,
                n_tracts=n_tr, slope_per_SD=b * unit, se=se * unit, t=t, df=df_, p=p,
                ci_lo=(b - half) * unit, ci_hi=(b + half) * unit, std_beta=b * unit / sd_y if sd_y > 0 else np.nan)


def slope_tests(long_df):
    out = []
    ca = long_df[long_df['cls'].isin(['commissural', 'association'])].reset_index(drop=True)
    sets = (('pooled_comm_assoc', ca), ('all_tracts', long_df))
    for metric in METRICS:
        for set_name, d in sets:
            if d['tract'].nunique() < 3 or d['subject'].nunique() < 10:
                continue
            tcode, scode = _codes(d)
            x = d[metric].to_numpy(float)
            y = d['effect'].to_numpy(float)
            unit = float(np.std(_demean(x, tcode, tcode.max() + 1), ddof=1))
            sd_y = float(np.std(_demean(y, tcode, tcode.max() + 1), ddof=1))
            for model, tw in (('M1_tractFE', False), ('M2_tract+participantFE', True)):
                b, V, df_, G = fe_ols(y, x[:, None], tcode, scode, tw)
                out.append(_row(model, set_name, metric, 'slope', b[0], float(np.sqrt(V[0, 0])), df_, unit, sd_y,
                                len(d), d['subject'].nunique(), d['tract'].nunique()))
        # class-specific slopes and their difference (one interaction model)
        if ca['cls'].nunique() == 2:
            tcode, scode = _codes(ca)
            x = ca[metric].to_numpy(float)
            y = ca['effect'].to_numpy(float)
            unit = float(np.std(_demean(x, tcode, tcode.max() + 1), ddof=1))
            sd_y = float(np.std(_demean(y, tcode, tcode.max() + 1), ddof=1))
            ic = (ca['cls'] == 'commissural').to_numpy(float)
            X = np.column_stack([x * ic, x * (1 - ic)])
            for model, tw in (('M1_tractFE', False), ('M2_tract+participantFE', True)):
                b, V, df_, G = fe_ols(y, X, tcode, scode, tw)
                for j, cname in enumerate(('commissural', 'association')):
                    d_ = ca[ca['cls'] == cname]
                    out.append(_row(model, cname, metric, 'slope', b[j], float(np.sqrt(V[j, j])), df_, unit, sd_y,
                                    len(d_), d_['subject'].nunique(), d_['tract'].nunique()))
                se_d = float(np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1]))
                out.append(_row(model, 'commissural_minus_association', metric, 'slope_difference', b[0] - b[1], se_d,
                                df_, unit, sd_y, len(ca), ca['subject'].nunique(), ca['tract'].nunique()))
    g = pd.DataFrame(out)
    g['family'] = g['model'] + '|' + g['term'] + '|' + np.where(g['set'].isin(['commissural', 'association']),
                                                                'class_specific', g['set'])
    g['q_bh'] = np.nan
    for fam, idx in g.groupby('family').groups.items():
        g.loc[idx, 'q_bh'] = bh(g.loc[idx, 'p'])
    return g


def per_tract_correlations(long_df):
    rows = []
    for (tract, cls), d in long_df.groupby(['tract', 'cls']):
        for metric in METRICS:
            if len(d) < MIN_ROWS_TRACT or d[metric].std() == 0 or d['effect'].std() == 0:
                continue
            r = float(np.corrcoef(d[metric], d['effect'])[0, 1])
            n = len(d)
            z = np.arctanh(np.clip(r, -0.9999, 0.9999))
            se = 1 / np.sqrt(n - 3)
            t = r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))
            rows.append(dict(tract=tract, cls=cls, metric=metric, n=n, r=r,
                             ci_lo=float(np.tanh(z - 1.96 * se)), ci_hi=float(np.tanh(z + 1.96 * se)),
                             p_unadjusted=float(2 * stats.t.sf(abs(t), n - 2))))
    return pd.DataFrame(rows)


def sign_test(corr):
    rows = []
    d = corr[corr['cls'].isin(['commissural', 'association'])]
    for metric, g in d.groupby('metric'):
        n_neg, n_pos = int((g['r'] < 0).sum()), int((g['r'] > 0).sum())
        p = stats.binomtest(n_neg, n_neg + n_pos, 0.5).pvalue if (n_neg + n_pos) > 0 else np.nan
        rows.append(dict(metric=metric, n_tracts=n_neg + n_pos, n_negative=n_neg, n_positive=n_pos,
                         p_binomial_two_sided=p))
    return pd.DataFrame(rows)


def leave_one_tract_out(long_df):
    ca = long_df[long_df['cls'].isin(['commissural', 'association'])]
    rows = []
    for metric in METRICS:
        tcode, _ = _codes(ca)
        unit = float(np.std(_demean(ca[metric].to_numpy(float), tcode, tcode.max() + 1), ddof=1))
        for model, tw in (('M1_tractFE', False), ('M2_tract+participantFE', True)):
            for tract in sorted(ca['tract'].unique()):
                d = ca[ca['tract'] != tract]
                tc, sc = _codes(d)
                b, V, df_, G = fe_ols(d['effect'].to_numpy(float), d[metric].to_numpy(float)[:, None], tc, sc, tw)
                se = float(np.sqrt(V[0, 0]))
                t = b[0] / se if se > 0 else np.nan
                rows.append(dict(model=model, metric=metric, left_out=tract, slope_per_SD=b[0] * unit,
                                 p=float(2 * stats.t.sf(abs(t), df_)) if np.isfinite(t) else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Whole analysis
# --------------------------------------------------------------------------
def analyse(long_df, skipped, knots, output_dir, make_figures=True):
    os.makedirs(output_dir, exist_ok=True)
    long_df.to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_participant_tract.csv'), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_skipped_participants.csv'),
                                     index=False)
    group = slope_tests(long_df)
    group.to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_group_stats.csv'), index=False)
    corr = per_tract_correlations(long_df)
    corr.to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_tract_corr.csv'), index=False)
    sign = sign_test(corr)
    sign.to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_tract_sign_test.csv'), index=False)
    loo = leave_one_tract_out(long_df)
    loo.to_csv(os.path.join(output_dir, 'aim1_wmstrength_within_leave_one_tract_out.csv'), index=False)
    ctx = long_df[['tract', 'subject']]
    with open(os.path.join(output_dir, 'aim1_wmstrength_within_summary.txt'), 'w') as f:
        f.write('Aim 1 extension: WM strength, within tract across participants '
                '(pt2_no03e_aim1_wm_strength_within_tract)\n')
        f.write(f'Participants analysed: {ctx["subject"].nunique()}; skipped: {len(skipped)}; '
                f'participant x tract rows: {len(long_df)}; tracts: {ctx["tract"].nunique()}\n')
        f.write('Spline knots (mm): ' + ', '.join(f'{np.exp(k):.1f}' for k in knots) + '\n\n')
        f.write('Slope = change of the WM effect (Fisher-z) per 1 SD of the within-tract deviation of the metric; '
                'CR1 SE clustered by participant.\n')
        cols = ['model', 'set', 'metric', 'term', 'n_rows', 'n_participants', 'n_tracts', 'slope_per_SD', 'ci_lo',
                'ci_hi', 'std_beta', 'p', 'q_bh']
        f.write(group[cols].to_string(index=False, float_format=lambda x: f'{x:.5g}') + '\n\n')
        f.write('Sign test across tracts (commissural + association; per-tract correlation across participants):\n')
        f.write(sign.to_string(index=False, float_format=lambda x: f'{x:.4g}') + '\n\n')
        piv = loo.groupby(['model', 'metric']).agg(min_slope=('slope_per_SD', 'min'), max_slope=('slope_per_SD', 'max'),
                                                   max_p=('p', 'max')).reset_index()
        f.write('Leave-one-tract-out range of the pooled slope:\n')
        f.write(piv.to_string(index=False, float_format=lambda x: f'{x:.5g}') + '\n')
    if make_figures:
        make_figure(group, corr, output_dir)
    return group, corr, sign, loo


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
FONT = 'Arial'


def _q_text(q):
    if not np.isfinite(q):
        return ''
    return 'q < .001' if q < 0.001 else f'q = {q:.3f}'.replace('0.', '.')


def make_figure(group, corr, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: the figure was skipped (rerun with --figures-only).')
        return
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'Liberation Sans', 'DejaVu Sans'],
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 7.5})
    fig = plt.figure(figsize=(7.4, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.35], left=0.17, right=0.90, top=0.89, bottom=0.07,
                          hspace=0.50, wspace=0.12)
    # ---- Panel A: forest of slopes
    axA = fig.add_subplot(gs[0, :])
    spec = [('M1_tractFE', 'pooled_comm_assoc', '#222222', 'Commissural + association, tract FE'),
            ('M2_tract+participantFE', 'pooled_comm_assoc', '#777777', 'Commissural + association, tract + participant FE'),
            ('M1_tractFE', 'commissural', CLASS_COLOUR['commissural'], 'Commissural, tract FE'),
            ('M1_tractFE', 'association', CLASS_COLOUR['association'], 'Association, tract FE')]
    y, yticks, ylabels = 0.0, [], []
    lo_all, hi_all = [], []
    for metric in METRICS:
        top = y
        for model, set_name, colour, label in spec:
            r = group[(group['model'] == model) & (group['set'] == set_name) & (group['metric'] == metric)
                      & (group['term'] == 'slope')]
            if not len(r):
                continue
            r = r.iloc[0]
            axA.plot([r['ci_lo'], r['ci_hi']], [y, y], color=colour, lw=1.2, solid_capstyle='butt')
            filled = model.startswith('M1')
            axA.plot([r['slope_per_SD']], [y], 'o', color=colour, ms=4.2, mec=colour if not filled else 'white',
                     mew=0.9 if not filled else 0.4, mfc=colour if filled else 'white', zorder=3)
            axA.text(1.01, y, _q_text(r['q_bh']), transform=axA.get_yaxis_transform(), fontsize=6.3, va='center')
            lo_all.append(r['ci_lo'])
            hi_all.append(r['ci_hi'])
            y -= 1.0
        yticks.append((top + y + 1.0) / 2)
        ylabels.append(metric)
        y -= 0.7
    axA.axvline(0, color='#555555', lw=0.6, ls=(0, (3, 2)))
    axA.set_yticks(yticks)
    axA.set_yticklabels(ylabels, fontsize=8.5, fontweight='bold')
    axA.tick_params(axis='y', length=0)
    pad = (max(hi_all) - min(lo_all)) * 0.12
    axA.set_xlim(min(lo_all) - pad, max(hi_all) + pad)
    axA.set_ylim(y + 0.3, 0.8)
    axA.spines[['top', 'right', 'left']].set_visible(False)
    axA.set_xlabel('Change in WM effect per 1 SD of within-tract deviation of the metric\n'
                   '(Fisher z; mean and 95% CI, cluster-robust by participant; filled = tract FE, open = + participant FE)',
                   fontsize=7, labelpad=2)
    fig.text(0.01, 0.975, 'A', fontsize=10, fontweight='bold')
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, marker='o', ms=4, lw=1.2, mfc=c if m.startswith('M1') else 'white', label=l)
               for m, s, c, l in spec]
    axA.legend(handles=handles, loc='lower left', bbox_to_anchor=(-0.28, 1.01), ncol=2, frameon=False, fontsize=6.3,
               columnspacing=1.2, handlelength=2.0)
    # ---- Panel B: per-tract correlations across participants
    d = corr[corr['cls'].isin(['commissural', 'association'])]
    order = (d[d['metric'] == 'FA'].sort_values(['cls', 'tract'])['tract'].tolist())
    yp = {t: i for i, t in enumerate(order[::-1])}
    for j, metric in enumerate(METRICS):
        ax = fig.add_subplot(gs[1, j])
        dm = d[d['metric'] == metric]
        for _, r in dm.iterrows():
            c = CLASS_COLOUR[r['cls']]
            ax.plot([r['ci_lo'], r['ci_hi']], [yp[r['tract']]] * 2, color=c, lw=0.8)
            ax.plot([r['r']], [yp[r['tract']]], 'o', color=c, ms=2.8)
        ax.axvline(0, color='#555555', lw=0.5, ls=(0, (3, 2)))
        ax.set_xlim(-0.35, 0.35)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.set_title(metric, fontsize=8, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=6, length=2.5, width=0.6)
        if j == 0:
            ax.set_yticks(list(yp.values()))
            ax.set_yticklabels(list(yp.keys()), fontsize=5.6)
        else:
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        ax.set_xlabel('r across participants', fontsize=6.5)
    y_title = ax.get_position().y1 + 0.04          # ax = last small panel of the bottom row
    fig.text(0.01, y_title, 'B', fontsize=10, fontweight='bold')
    fig.text(0.04, y_title, 'Within-tract correlation of the metric and the WM effect (per tract, 95% CI)',
             fontsize=7.5, fontweight='bold')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(os.path.join(output_dir, f'fig_aim1_wm_strength_within.{ext}'), dpi=FIG_DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Self-test (synthetic data only)
# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(3)
    n_sub, n_tr = 300, 14
    tracts = [(f'cc.t{i}', 'commissural') for i in range(5)] + [(f'lh.a{i}', 'association') for i in range(n_tr - 5)]
    tract_mean = {t: rng.uniform(0.35, 0.6) for t, _ in tracts}
    tract_eff = {t: rng.uniform(-0.01, 0.05) for t, _ in tracts}
    rows = []
    for s in range(n_sub):
        u = rng.normal(0, 1)                         # participant-level factor (signal quality, site, ...)
        for t, cls in tracts:
            dev = rng.normal(0, 0.03)
            fa = tract_mean[t] + dev
            eff = tract_eff[t] + 0.01 * u + 0.012 * (dev / 0.03) + rng.normal(0, 0.02)   # true within-tract FA effect
            md = 8e-4 + 1e-5 * u + rng.normal(0, 3e-5)                                  # confounded by u only
            ad = 1.2e-3 + rng.normal(0, 3e-5)                                           # unrelated
            rd = 5.5e-4 - 2e-5 * (dev / 0.03) + rng.normal(0, 2e-5)                    # follows FA
            rows.append(dict(subject=f'{1000 + s}', tract=t, cls=cls, effect=eff, n_pairs=3,
                             FA=fa, MD=md, AD=ad, RD=rd))
    long_df = pd.DataFrame(rows)
    out = os.path.join(tempfile.mkdtemp(prefix='wmstrength_within_selftest_'), 'out')
    group, corr, sign, loo = analyse(long_df, [], np.log([30, 60, 90, 120, 150]), out, make_figures=True)
    g = group.set_index(['model', 'set', 'metric', 'term'])
    fa1 = g.loc[('M1_tractFE', 'pooled_comm_assoc', 'FA', 'slope')]
    fa2 = g.loc[('M2_tract+participantFE', 'pooled_comm_assoc', 'FA', 'slope')]
    md1 = g.loc[('M1_tractFE', 'pooled_comm_assoc', 'MD', 'slope')]
    md2 = g.loc[('M2_tract+participantFE', 'pooled_comm_assoc', 'MD', 'slope')]
    ad1 = g.loc[('M1_tractFE', 'pooled_comm_assoc', 'AD', 'slope')]
    assert fa1['slope_per_SD'] > 0.008 and fa1['p'] < 1e-6, fa1['slope_per_SD']      # true effect found
    assert fa2['slope_per_SD'] > 0.008 and fa2['p'] < 1e-6, fa2['slope_per_SD']      # survives participant FE
    assert md1['slope_per_SD'] > 0.003 and md1['p'] < 1e-3, md1['slope_per_SD']      # confounded: M1 is fooled
    assert abs(md2['slope_per_SD']) < 0.003 and md2['p'] > 0.01, md2                 # M2 removes the confound
    assert ad1['p'] > 0.001, ad1['p']                                               # unrelated metric
    assert (sign['n_tracts'] == n_tr).all()
    for name in ('aim1_wmstrength_within_group_stats.csv', 'aim1_wmstrength_within_tract_corr.csv',
                 'aim1_wmstrength_within_summary.txt'):
        assert os.path.isfile(os.path.join(out, name)), name
    print(f'Self-test passed (synthetic data). FA slope M1 {fa1["slope_per_SD"]:.4f}, M2 {fa2["slope_per_SD"]:.4f}; '
          f'confounded MD slope M1 {md1["slope_per_SD"]:.4f} -> M2 {md2["slope_per_SD"]:.4f}. Output: {out}')


def figures_only(output_dir):
    group = pd.read_csv(os.path.join(output_dir, 'aim1_wmstrength_within_group_stats.csv'))
    corr = pd.read_csv(os.path.join(output_dir, 'aim1_wmstrength_within_tract_corr.csv'))
    make_figure(group, corr, output_dir)
    print(f'Figures rewritten in {output_dir}')


def main():
    ap = argparse.ArgumentParser(description='Optional exploratory analysis: WM strength within tract across participants')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--figures-only', action='store_true')
    ap.add_argument('--input', default=INPUT_CSV)
    ap.add_argument('--tract-level', default=TRACT_LEVEL_CSV)
    ap.add_argument('--batch-dir', default=BATCH_DIR)
    ap.add_argument('--output', default=OUTPUT_DIR)
    ap.add_argument('--cortical-categories', nargs='+', default=sorted(CORTICAL_CATEGORIES))
    ap.add_argument('--max-subjects', type=int, default=None, help='testing only')
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.figures_only:
        figures_only(args.output)
        return
    cats = set(args.cortical_categories)
    df = load_pairs(args.input, cats, args.max_subjects)
    tracts = load_tracts(args.tract_level, cats)
    long_df, skipped, knots = build_long_table(df, tracts, args.batch_dir)
    group, _, sign, _ = analyse(long_df, skipped, knots, args.output)
    print(f'Participants analysed: {long_df["subject"].nunique()}; skipped: {len(skipped)}')
    if skipped:
        print(pd.DataFrame(skipped)['reason'].value_counts().to_string())
    cols = ['model', 'set', 'metric', 'term', 'n_participants', 'slope_per_SD', 'ci_lo', 'ci_hi', 'p', 'q_bh']
    print(group[group['set'].isin(['pooled_comm_assoc', 'commissural_minus_association'])][cols].to_string(index=False))
    print(sign.to_string(index=False))
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
