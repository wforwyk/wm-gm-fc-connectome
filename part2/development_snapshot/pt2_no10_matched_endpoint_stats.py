#!/usr/bin/env python3
# =============================================================================
# pt2_no10_matched_endpoint_stats.py
#
# AIM 2 (CONFOUNDER-CONTROLLED) / PURPOSE:
#   Perform group-level inference using depth/GM-probability/tSNR-matched
#   endpoint controls matched for WM distance and GM probability.  The primary test is the within-subject endpoint ×
#   target-type interaction.  A covariate-adjusted subject-fixed-effect model
#   is also fitted as a sensitivity analysis when statsmodels is available.
# INPUT: pt2_no5 v2 subject CSVs and matching-balance CSVs.
# OUTPUT: CSV/txt summaries, subject-level contrasts, matching QC summary.
# USER SETTINGS: edit only the section immediately below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no5_matched_endpoint_fc_v2'
OUTPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no6_matched_endpoint_stats_v2'
MAX_ABSOLUTE_SMD=0.10  # conventional balance target; report but do not drop data
# =============================================================================

def one_sample(x,label):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    if len(x)<2:return {'test':label,'n_subjects':len(x)}
    t,p=stats.ttest_1samp(x,0);se=x.std(ddof=1)/np.sqrt(len(x));ci=stats.t.ppf(.975,len(x)-1)*se
    return {'test':label,'n_subjects':len(x),'mean':x.mean(),'se':se,'ci95_lo':x.mean()-ci,'ci95_hi':x.mean()+ci,'t':t,'df':len(x)-1,'p':p}
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True); files=glob.glob(os.path.join(INPUT_DIR,'*_aim2_matched_endpoint_fc.csv'))
    if not files:raise RuntimeError('No pt2_no5 v2 files found.')
    d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).dropna(subset=['mean_fc_z'])
    # Average multiple adjacent targets within tract end, then average tract ends within subject.
    u=d.groupby(['subject','tract_name','endpoint','source_roi','voxel_class','target_type'],as_index=False).agg(mean_fc_z=('mean_fc_z','mean'),fc_distance_log_slope=('fc_distance_log_slope','mean'),wm_boundary_distance=('wm_boundary_distance','mean'),gm_probability=('gm_probability','mean'))
    w=u.pivot_table(index=['subject','tract_name','endpoint','source_roi','voxel_class'],columns='target_type',values='mean_fc_z').reset_index();w['distant_minus_adjacent']=w['wm_connected_distant']-w['physically_adjacent']
    s=w.groupby(['subject','voxel_class']).distant_minus_adjacent.mean().unstack().dropna();s['endpoint_specific_distant_preference']=s['endpoint']-s['matched_non_endpoint'];primary=one_sample(s.endpoint_specific_distant_preference,'Aim 2.1 matched endpoint × distant-vs-adjacent interaction')
    endpoint_control=[]
    for target,g in u.groupby('target_type'):
        q=g.groupby(['subject','voxel_class']).mean_fc_z.mean().unstack().dropna(); r=one_sample(q['endpoint']-q['matched_non_endpoint'],f'Aim 2.2 matched endpoint-control, {target}');r['target_type']=target;endpoint_control.append(r)
    slope=u.groupby(['subject','voxel_class']).fc_distance_log_slope.mean().unstack().dropna();slope_result=one_sample(slope['endpoint']-slope['matched_non_endpoint'],'Aim 2.3 matched endpoint-control FC~log(distance) slope')
    balance_files=glob.glob(os.path.join(INPUT_DIR,'*_aim2_matching_balance.csv')); balance=pd.concat([pd.read_csv(f) for f in balance_files],ignore_index=True) if balance_files else pd.DataFrame();
    if len(balance):
        cols=['smd_wm_distance','smd_gm_probability'];balance['all_covariates_balanced']=balance[cols].abs().max(axis=1)<=MAX_ABSOLUTE_SMD;balance.to_csv(os.path.join(OUTPUT_DIR,'aim2_matching_balance_all.csv'),index=False)
    model_result='statsmodels unavailable; primary matched subject-level analysis remains valid.'
    try:
        import statsmodels.formula.api as smf
        m=d.copy();m['endpoint_binary']=(m.voxel_class=='endpoint').astype(int);m['distant_binary']=(m.target_type=='wm_connected_distant').astype(int)
        fit=smf.ols('mean_fc_z ~ endpoint_binary*distant_binary + wm_boundary_distance + gm_probability + C(subject)',data=m).fit(cov_type='cluster',cov_kwds={'groups':m['subject']})
        model_result=fit.summary().as_text();pd.DataFrame({'term':fit.params.index,'beta':fit.params.values,'se_cluster_subject':fit.bse.values,'p':fit.pvalues.values}).to_csv(os.path.join(OUTPUT_DIR,'aim2_covariate_adjusted_model.csv'),index=False)
    except Exception as e:model_result=f'Covariate-adjusted model not fitted: {e}'
    s.to_csv(os.path.join(OUTPUT_DIR,'aim2_subject_interaction.csv'));w.to_csv(os.path.join(OUTPUT_DIR,'aim2_unit_contrasts.csv'),index=False);pd.DataFrame([primary]).to_csv(os.path.join(OUTPUT_DIR,'aim2_primary_stats.csv'),index=False);pd.DataFrame(endpoint_control).to_csv(os.path.join(OUTPUT_DIR,'aim2_endpoint_control_stats.csv'),index=False);pd.DataFrame([slope_result]).to_csv(os.path.join(OUTPUT_DIR,'aim2_distance_slope_stats.csv'),index=False)
    with open(os.path.join(OUTPUT_DIR,'aim2_stats.txt'),'w') as f:f.write(pd.DataFrame([primary]+endpoint_control+[slope_result]).to_string(index=False)+'\n\nCOVARIATE-ADJUSTED SENSITIVITY MODEL\n'+model_result)
if __name__=='__main__':main()
