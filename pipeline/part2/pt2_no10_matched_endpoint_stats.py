#!/usr/bin/env python3
# =============================================================================
# pt2_no10_matched_endpoint_stats.py
#
# AIM 2 (CONFOUNDER-CONTROLLED) / PURPOSE:
#   Perform group-level inference using WM-distance/GM-probability-matched
#   endpoint controls matched for WM distance and GM probability.  The primary test is the within-subject endpoint ×
#   target-type interaction.  A covariate-adjusted subject-fixed-effect model
#   is also fitted as a sensitivity analysis when statsmodels is available.
# INPUT: pt2_no9 subject CSVs and matching-balance CSVs.
# OUTPUT: CSV/txt summaries, subject-level contrasts, balance-all and
# balance-summary QC tables, and an FDR-adjusted target-contrast table.
# USER SETTINGS: edit only the section immediately below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR='/path/to/your/project/derivatives/pt2_group_results/no9_matched_endpoint_fc'
OUTPUT_DIR='/path/to/your/project/derivatives/pt2_group_results/no10_matched_endpoint_stats'
MAX_ABSOLUTE_SMD=0.10
MIN_MATCHED_VOXELS=10
REQUIRE_BALANCE=True
# =============================================================================

def one_sample(x,label):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<2:return {'test':label,'n_subjects':len(x)}
    t,p=stats.ttest_1samp(x,0);se=x.std(ddof=1)/np.sqrt(len(x));ci=stats.t.ppf(.975,len(x)-1)*se
    return {'test':label,'n_subjects':len(x),'mean':x.mean(),'se':se,'ci95_lo':x.mean()-ci,'ci95_hi':x.mean()+ci,'t':t,'df':len(x)-1,'p':p}
def benjamini_hochberg(pvals):
    pvals=np.asarray(pvals,float);out=np.full(len(pvals),np.nan);valid=np.isfinite(pvals)
    if not valid.any(): return out
    p=pvals[valid];order=np.argsort(p);ranked=p[order];adj=np.minimum.accumulate((ranked*len(ranked)/np.arange(1,len(ranked)+1))[::-1])[::-1]
    restored=np.empty_like(adj);restored[order]=np.minimum(adj,1.0);out[valid]=restored;return out
def read_nonempty_csvs(files):
    """Read only subject files containing rows; no-match subjects are expected."""
    frames=[]
    for file in files:
        try:
            frame=pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty: frames.append(frame)
    return frames
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True); files=glob.glob(os.path.join(INPUT_DIR,'*_aim2_matched_endpoint_fc.csv'))
    if not files:raise RuntimeError('No pt2_no9 matched FC files found.')
    data_frames=read_nonempty_csvs(files)
    if not data_frames:
        raise RuntimeError('No matched FC rows were produced. Inspect no09 matching-balance CSVs: the endpoint definition and matching criteria left no analysable tract endpoints.')
    d=pd.concat(data_frames,ignore_index=True).dropna(subset=['mean_fc_z'])
    if d.empty:
        raise RuntimeError('Matched FC rows contain no finite mean_fc_z values.')
    required_data_columns={'mean_distance_mm','endpoint_relative_threshold'}
    missing_data_columns=required_data_columns-set(d.columns)
    if missing_data_columns:
        raise RuntimeError(f'Matched FC input lacks {sorted(missing_data_columns)}; rerun pt2_no08 and pt2_no09.')
    thresholds=np.unique(d.endpoint_relative_threshold.dropna().to_numpy(float))
    if len(thresholds)!=1:
        raise RuntimeError(f'Matched FC input combines endpoint thresholds {thresholds.tolist()}; analyse one threshold lineage at a time.')
    endpoint_relative_threshold=float(thresholds[0])
    balance_files=glob.glob(os.path.join(INPUT_DIR,'*_aim2_matching_balance.csv'))
    balance_frames=read_nonempty_csvs(balance_files)
    balance=pd.concat(balance_frames,ignore_index=True) if balance_frames else pd.DataFrame()
    if REQUIRE_BALANCE and balance.empty:
        raise RuntimeError('Matching-balance CSVs are required for the primary matched analysis but were not found.')
    if len(balance):
        cols=['smd_wm_distance','smd_gm_probability']
        missing=[c for c in ['subject','tract','endpoint','n_matched',*cols] if c not in balance]
        if missing: raise RuntimeError(f'Matching-balance CSV is missing required columns: {missing}')
        balance['all_covariates_balanced']=balance[cols].abs().max(axis=1)<=MAX_ABSOLUTE_SMD
        balance['eligible_primary']=balance.n_matched.ge(MIN_MATCHED_VOXELS) & (balance.all_covariates_balanced if REQUIRE_BALANCE else True)
        eligible=balance.loc[balance.eligible_primary,['subject','tract','endpoint']].rename(columns={'tract':'tract_name'})
        if eligible.empty: raise RuntimeError('No tract endpoints passed the matched-analysis balance criteria.')
        d=d.merge(eligible,on=['subject','tract_name','endpoint'],how='inner')
        if d.empty: raise RuntimeError('No matched FC rows remained after applying balance criteria.')
        balance.to_csv(os.path.join(OUTPUT_DIR,'aim2_matching_balance_all.csv'),index=False)
        summary=pd.DataFrame([{
            'n_subjects_total':balance.subject.nunique(), 'n_endpoint_units_total':len(balance),
            'n_endpoint_units_eligible':int(balance.eligible_primary.sum()),
            'eligible_endpoint_pct':100*balance.eligible_primary.mean(),
            'median_matched_voxels_all':balance.n_matched.median(),
            'median_matched_voxels_eligible':balance.loc[balance.eligible_primary,'n_matched'].median(),
            'median_abs_smd_wm_distance_eligible':balance.loc[balance.eligible_primary,'smd_wm_distance'].abs().median(),
            'median_abs_smd_gm_probability_eligible':balance.loc[balance.eligible_primary,'smd_gm_probability'].abs().median(),
            'max_absolute_smd_threshold':MAX_ABSOLUTE_SMD, 'minimum_matched_voxels':MIN_MATCHED_VOXELS,
            'endpoint_relative_threshold':endpoint_relative_threshold,
        }])
        summary.to_csv(os.path.join(OUTPUT_DIR,'aim2_matching_balance_summary.csv'),index=False)
    # Average multiple adjacent targets within tract end, then average tract ends within subject.
    u=d.groupby(['subject','tract_name','endpoint','source_roi','voxel_class','target_type'],as_index=False).agg(mean_fc_z=('mean_fc_z','mean'),fc_distance_log_slope=('fc_distance_log_slope','mean'),wm_boundary_distance=('wm_boundary_distance','mean'),gm_probability=('gm_probability','mean'))
    w=u.pivot_table(index=['subject','tract_name','endpoint','source_roi','voxel_class'],columns='target_type',values='mean_fc_z').reset_index()
    required_targets={'wm_connected_distant','physically_adjacent'}
    if not required_targets.issubset(w.columns): raise RuntimeError(f'Matched FC data lack required targets: {required_targets-set(w.columns)}')
    w['distant_minus_adjacent']=w['wm_connected_distant']-w['physically_adjacent']
    s=w.groupby(['subject','voxel_class']).distant_minus_adjacent.mean().unstack().dropna();s['endpoint_specific_distant_preference']=s['endpoint']-s['matched_non_endpoint'];primary=one_sample(s.endpoint_specific_distant_preference,'Aim 2.1 matched endpoint × distant-vs-adjacent interaction')
    endpoint_control=[]
    for target,g in u.groupby('target_type'):
        q=g.groupby(['subject','voxel_class']).mean_fc_z.mean().unstack().dropna(); r=one_sample(q['endpoint']-q['matched_non_endpoint'],f'Aim 2.2 matched endpoint-control, {target}');r['target_type']=target;endpoint_control.append(r)
    if endpoint_control:
        pvals=[r.get('p',np.nan) for r in endpoint_control]
        for r,padj in zip(endpoint_control,benjamini_hochberg(pvals)): r['p_fdr_bh']=padj
    slope=u.groupby(['subject','voxel_class']).fc_distance_log_slope.mean().unstack().dropna();slope_result=one_sample(slope['endpoint']-slope['matched_non_endpoint'],'Aim 2.3 matched endpoint-control FC~log(distance) slope')
    # Directly quantify whether endpoint/control contrasts differ in geodesic
    # distance to each target, then include log target distance in the
    # covariate-adjusted interaction sensitivity model.
    distance_unit=(d.groupby(['subject','tract_name','endpoint','source_roi','target_roi','target_type','voxel_class'],as_index=False)
                     .agg(mean_target_geodesic_distance_mm=('mean_distance_mm','mean')))
    distance_wide=(distance_unit.pivot_table(index=['subject','tract_name','endpoint','source_roi','target_roi','target_type'],columns='voxel_class',values='mean_target_geodesic_distance_mm')
                   .rename(columns={'endpoint':'endpoint_target_distance_mm','matched_non_endpoint':'control_target_distance_mm'})
                   .reset_index())
    if {'endpoint_target_distance_mm','control_target_distance_mm'}.issubset(distance_wide.columns):
        distance_wide['endpoint_minus_control_target_distance_mm']=distance_wide['endpoint_target_distance_mm']-distance_wide['control_target_distance_mm']
        distance_subject=(distance_wide.groupby(['subject','target_type'],as_index=False)
                          .endpoint_minus_control_target_distance_mm.mean())
        distance_results=[]
        for target,g in distance_subject.groupby('target_type'):
            r=one_sample(g.endpoint_minus_control_target_distance_mm,
                         f'Aim 2 target-distance check, endpoint-control, {target}')
            r['target_type']=target
            distance_results.append(r)
        pd.DataFrame(distance_results).to_csv(os.path.join(OUTPUT_DIR,'aim2_target_distance_balance_stats.csv'),index=False)
        distance_wide.to_csv(os.path.join(OUTPUT_DIR,'aim2_target_distance_balance_units.csv'),index=False)
    else:
        raise RuntimeError('Could not form paired endpoint/control target-distance QC table.')

    model_result='statsmodels unavailable; primary matched subject-level analysis remains valid.'
    try:
        import statsmodels.formula.api as smf
        m=d[np.isfinite(d.mean_distance_mm) & d.mean_distance_mm.gt(0)].copy()
        m['endpoint_binary']=(m.voxel_class=='endpoint').astype(int)
        m['distant_binary']=(m.target_type=='wm_connected_distant').astype(int)
        m['log_target_geodesic_distance']=np.log(m.mean_distance_mm)
        fit=smf.ols('mean_fc_z ~ endpoint_binary*distant_binary + log_target_geodesic_distance + wm_boundary_distance + gm_probability + C(subject)',data=m).fit(cov_type='cluster',cov_kwds={'groups':m['subject']})
        model_result=fit.summary().as_text()
        model_table=pd.DataFrame({'term':fit.params.index,'beta':fit.params.values,'se_cluster_subject':fit.bse.values,'p':fit.pvalues.values})
        model_table.to_csv(os.path.join(OUTPUT_DIR,'aim2_covariate_adjusted_model.csv'),index=False)
        model_table.to_csv(os.path.join(OUTPUT_DIR,'aim2_target_distance_adjusted_model.csv'),index=False)
    except Exception as e:model_result=f'Covariate-adjusted model not fitted: {e}'
    sample=s.reset_index()[['subject']].drop_duplicates();sample.to_csv(os.path.join(OUTPUT_DIR,'aim2_analysis_subjects.csv'),index=False)
    s.to_csv(os.path.join(OUTPUT_DIR,'aim2_subject_interaction.csv'));w.to_csv(os.path.join(OUTPUT_DIR,'aim2_unit_contrasts.csv'),index=False);pd.DataFrame([primary]).to_csv(os.path.join(OUTPUT_DIR,'aim2_primary_stats.csv'),index=False);pd.DataFrame(endpoint_control).to_csv(os.path.join(OUTPUT_DIR,'aim2_endpoint_control_stats.csv'),index=False);pd.DataFrame([slope_result]).to_csv(os.path.join(OUTPUT_DIR,'aim2_distance_slope_stats.csv'),index=False)
    with open(os.path.join(OUTPUT_DIR,'aim2_stats.txt'),'w') as f:f.write(pd.DataFrame([primary]+endpoint_control+[slope_result]).to_string(index=False)+f'\n\nEndpoint relative PD threshold: {endpoint_relative_threshold:g}\n\nTARGET GEODESIC-DISTANCE QC\n'+pd.DataFrame(distance_results).to_string(index=False)+'\n\nCOVARIATE- AND TARGET-DISTANCE-ADJUSTED SENSITIVITY MODEL\n'+model_result)
if __name__=='__main__':main()
