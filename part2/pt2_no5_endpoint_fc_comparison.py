#!/usr/bin/env python3
# =============================================================================
# pt2_no5_endpoint_fc_comparison.py
#
# AIM 2 / PURPOSE:
#   Reuse Aim 1 FC/distance matrices to compare endpoint voxels against matched
#   non-endpoint voxels from the SAME source AAL region.  Both are tested for
#   FC to the tract's WM-connected distant region and to adjacent AAL regions.
# INPUT: pt2_no4 labels; Aim 1 BnB/AAL images; *_R.npz; *_W_a.npz; adjacency CSV.
# OUTPUT: {subj}_aim2_endpoint_fc.csv, one row per tract end/control/target.
# USER SETTINGS: edit only the block below.
# =============================================================================
import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

# =============================================================================
# USER SETTINGS
# =============================================================================
BATCH_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch'
TEMPLATE_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/template/AAL3'
LABEL_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no4_endpoint_voxels'
OUTPUT_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no5_endpoint_fc'
N_CONTROL_REPEATS = 10
RANDOM_SEED = 20260728
SUBJECT_LIST = []
# =============================================================================

def positions(src, tgt, n):
    i=np.minimum(np.repeat(src,len(tgt)), np.tile(tgt,len(src))); j=np.maximum(np.repeat(src,len(tgt)), np.tile(tgt,len(src)))
    valid=i<j; return (i[valid]*n-i[valid]*(i[valid]+1)//2+j[valid]-i[valid]-1).astype(np.int64)

def summary(src, tgt, n, fc, dist):
    pos=positions(src,tgt,n); f=fc[pos].astype(float); d=dist[pos].astype(float); keep=np.isfinite(f)&np.isfinite(d)
    if not keep.any(): return np.nan,np.nan,np.nan,0
    # Mean Fisher Z is used for inference; mean r is descriptive only.
    return float(np.mean(f[keep])),float(np.mean(np.arctanh(np.clip(f[keep],-.9999,.9999)))),float(np.mean(d[keep])),int(keep.sum())

def distance_slope(src, targets, n, fc, dist):
    """Within-source-voxel-set FC~log(geodesic distance) slope for Aim 2.3."""
    pos=positions(src,targets,n); f=fc[pos].astype(float); d=dist[pos].astype(float)
    keep=np.isfinite(f)&np.isfinite(d)&(d>0)&(np.abs(f)<.9999)
    if keep.sum()<10: return np.nan, np.nan, int(keep.sum())
    fit=stats.linregress(np.log(d[keep]),np.arctanh(f[keep])); return float(fit.slope),float(fit.rvalue),int(keep.sum())

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True); rng=np.random.default_rng(RANDOM_SEED)
    adj=pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_adjacent_pairs.csv')).query('both_included == 1')
    neighbors={}
    for r in adj.itertuples(index=False): neighbors.setdefault(r.roi_id_a,set()).add(r.roi_id_b); neighbors.setdefault(r.roi_id_b,set()).add(r.roi_id_a)
    subjects=SUBJECT_LIST or sorted(f.split('_')[0] for f in os.listdir(LABEL_DIR) if f.endswith('_endpoint_voxel_labels.npz'))
    for subj in subjects:
        label_path=os.path.join(LABEL_DIR,f'{subj}_endpoint_voxel_labels.npz')
        if not os.path.exists(label_path): continue
        z=np.load(label_path,allow_pickle=True); base=os.path.join(BATCH_DIR,subj)
        bnb=nib.load(os.path.join(base,'gm','func','derived',f'bnbmdenoised_detrended_rest_{subj}.nii')); d=bnb.get_fdata(); gm=d[...,0]>0 if d.ndim==4 else d>0; n=int(gm.sum())
        aal=nib.load(os.path.join(base,'gm','derived',f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii')).get_fdata().astype(int)[gm]
        regions={r:np.flatnonzero(aal==r) for r in np.unique(aal) if r>0}
        R=np.load(os.path.join(base,'gm','func','derived','fc_results',f'{subj}_R.npz'))['arr_0']; W=np.load(os.path.join(base,'gm','anat','derived','dist_results',f'{subj}_W_a.npz'))['arr_0']; ii,jj=np.triu_indices(n,1); fc=R[ii,jj] if R.ndim==2 else R; dist=W[ii,jj] if W.ndim==2 else W
        rows=[]
        for k,tract in enumerate(z['tract']):
            source=int(z['source_roi'][k]); remote=int(z['remote_roi'][k]); endpoint=z['endpoint_indices'][k]; candidates=z['control_candidates'][k]
            targets=[('wm_connected_distant',remote)]+[('physically_adjacent',x) for x in sorted(neighbors.get(source,set())-{remote}) if x in regions]
            if remote not in regions or len(candidates)<len(endpoint): continue
            distance_targets=np.concatenate([idx for roi,idx in regions.items() if roi != source])
            for repeat in range(N_CONTROL_REPEATS+1):
                cls='endpoint' if repeat==0 else 'non_endpoint_control'; src=endpoint if repeat==0 else rng.choice(candidates,len(endpoint),replace=False)
                slope,slope_r,slope_n=distance_slope(src,distance_targets,n,fc,dist)
                for target_type,target in targets:
                    mr,mz,md,npair=summary(src,regions[target],n,fc,dist)
                    rows.append({'subject':subj,'tract_name':tract,'endpoint':int(z['endpoint'][k]),'source_roi':source,'target_roi':target,'target_type':target_type,'voxel_class':cls,'control_repeat':repeat,'mean_fc_r':mr,'mean_fc_z':mz,'mean_distance_mm':md,'n_source_voxels':len(src),'n_voxel_pairs':npair,'fc_distance_log_slope':slope,'fc_distance_r':slope_r,'fc_distance_n_pairs':slope_n})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,f'{subj}_aim2_endpoint_fc.csv'),index=False); print(f'{subj}: {len(rows)} rows')

if __name__ == '__main__': main()
