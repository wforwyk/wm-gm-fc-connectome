#!/usr/bin/env python3
# =============================================================================
# pt2_no6_endpoint_stats.py
#
# AIM 2 / PURPOSE:
#   Aggregate pt2_no5 output with SUBJECT as the independent unit, then test:
#   2.1 endpoint-specific FC preference for WM-connected distant versus AAL-
#       adjacent targets (endpoint/control interaction);
#   2.2 endpoint versus non-endpoint FC for each target type;
#   2.3 endpoint versus control difference in FC~log(geodesic distance) slope.
# INPUT: per-subject {subj}_aim2_endpoint_fc.csv files from pt2_no5.
# OUTPUT: Aim 2 subject contrasts and plain-text/csv statistical summaries.
# USER SETTINGS: edit only the block below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no5_endpoint_fc'
OUTPUT_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no6_endpoint_stats'
# =============================================================================

def test(x, label):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2: return {'test':label,'n_subjects':len(x)}
    t,p=stats.ttest_1samp(x,0); se=x.std(ddof=1)/np.sqrt(len(x)); ci=stats.t.ppf(.975,len(x)-1)*se
    return {'test':label,'n_subjects':len(x),'mean':x.mean(),'se':se,'ci95_lo':x.mean()-ci,'ci95_hi':x.mean()+ci,'t':t,'df':len(x)-1,'p':p}

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    files=sorted(glob.glob(os.path.join(INPUT_DIR,'*_aim2_endpoint_fc.csv')))
    if not files: raise RuntimeError(f'No pt2_no5 CSVs found in {INPUT_DIR}')
    d=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).dropna(subset=['mean_fc_z'])
    # Average adjacent regions and repeated control samples within each tract end.
    u=(d.groupby(['subject','tract_name','endpoint','source_roi','voxel_class','target_type'],as_index=False)
         .agg(mean_fc_z=('mean_fc_z','mean'),fc_distance_log_slope=('fc_distance_log_slope','mean')))
    wide=u.pivot_table(index=['subject','tract_name','endpoint','source_roi','voxel_class'],columns='target_type',values='mean_fc_z',aggfunc='mean').reset_index()
    wide['distant_minus_adjacent']=wide['wm_connected_distant']-wide['physically_adjacent']
    subject=wide.groupby(['subject','voxel_class'],as_index=False).distant_minus_adjacent.mean().pivot(index='subject',columns='voxel_class',values='distant_minus_adjacent').dropna()
    subject['endpoint_specific_distant_preference']=subject['endpoint']-subject['non_endpoint_control']
    primary=test(subject.endpoint_specific_distant_preference,'Aim 2.1: (endpoint distant-adjacent) - (control distant-adjacent)')
    target=[]
    for name,g in u.groupby('target_type'):
        x=g.groupby(['subject','voxel_class']).mean_fc_z.mean().unstack().dropna(); r=test(x.endpoint-x.non_endpoint_control,f'Aim 2.2: endpoint-control FC, {name}'); r['target_type']=name; target.append(r)
    slopes=u.groupby(['subject','voxel_class']).fc_distance_log_slope.mean().unstack().dropna()
    slope=test(slopes.endpoint-slopes.non_endpoint_control,'Aim 2.3: endpoint-control FC~log(distance) slope')
    wide.to_csv(os.path.join(OUTPUT_DIR,'aim2_unit_contrasts.csv'),index=False); subject.to_csv(os.path.join(OUTPUT_DIR,'aim2_subject_interaction.csv'))
    pd.DataFrame([primary]).to_csv(os.path.join(OUTPUT_DIR,'aim2_primary_stats.csv'),index=False); pd.DataFrame(target).to_csv(os.path.join(OUTPUT_DIR,'aim2_endpoint_control_stats.csv'),index=False); pd.DataFrame([slope]).to_csv(os.path.join(OUTPUT_DIR,'aim2_distance_slope_stats.csv'),index=False)
    with open(os.path.join(OUTPUT_DIR,'aim2_stats.txt'),'w') as f: f.write(pd.DataFrame([primary]+target+[slope]).to_string(index=False))

if __name__ == '__main__': main()
