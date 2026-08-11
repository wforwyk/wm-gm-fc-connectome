#!/usr/bin/env python3
# =============================================================================
# pt2_no5_matched_endpoint_fc_v2.py
#
# AIM 2 (CONFOUNDER-CONTROLLED) / PURPOSE:
#   Match every endpoint voxel to one non-endpoint voxel in the same AAL region
#   using WM-boundary distance, GM probability, and tSNR.  Compare matched sets'
#   FC to WM-connected distant versus physically adjacent AAL regions.
# INPUT: pt2_no4 v2 NPZ; Aim 1 BnB/AAL images; R and W matrices; adjacency CSV.
# OUTPUT: {subj}_aim2_matched_endpoint_fc.csv; matching-quality CSV.
# USER SETTINGS: edit only the section immediately below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

# =============================================================================
# USER SETTINGS
# =============================================================================
BATCH_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch'
TEMPLATE_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/template/AAL3'
INPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no4_endpoint_covariates_v2'
OUTPUT_DIR='/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/pt2_group_results/no5_matched_endpoint_fc_v2'
MATCH_CALIPER=2.0  # maximum standardized Euclidean distance; set None to disable
RANDOM_SEED=20260729
SUBJECT_LIST=[]
# =============================================================================

def upper(src,tgt,n):
    i=np.minimum(np.repeat(src,len(tgt)),np.tile(tgt,len(src)));j=np.maximum(np.repeat(src,len(tgt)),np.tile(tgt,len(src)));good=i<j;return (i[good]*n-i[good]*(i[good]+1)//2+j[good]-i[good]-1).astype(np.int64)
def fc_summary(src,tgt,n,R,W):
    p=upper(src,tgt,n);r=R[p].astype(float);w=W[p].astype(float);good=np.isfinite(r)&np.isfinite(w)
    return (float(np.mean(np.arctanh(np.clip(r[good],-.9999,.9999)))),float(np.mean(w[good])),int(good.sum())) if good.any() else (np.nan,np.nan,0)
def distance_slope(src,tgt,n,R,W):
    p=upper(src,tgt,n);r=R[p].astype(float);w=W[p].astype(float);good=np.isfinite(r)&np.isfinite(w)&(w>0)&(np.abs(r)<.9999)
    if good.sum()<10:return np.nan
    return float(stats.linregress(np.log(w[good]),np.arctanh(r[good])).slope)
def smd(x,y):
    den=np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2);return float((np.mean(x)-np.mean(y))/den) if den>0 else np.nan
def match(endpoint,candidates,cov,rng):
    good=candidates[np.all(np.isfinite(cov[candidates]),axis=1)]; endpoint=endpoint[np.all(np.isfinite(cov[endpoint]),axis=1)]
    if len(good)<len(endpoint): return endpoint,np.array([],dtype=int),np.array([])
    allx=np.vstack([cov[endpoint],cov[good]]); mu=allx.mean(0); sd=allx.std(0); sd[sd==0]=1; e=(cov[endpoint]-mu)/sd; c=(cov[good]-mu)/sd; available=np.ones(len(good),bool); chosen=[]; distances=[]
    for ix in rng.permutation(len(endpoint)):
        ds=np.sqrt(((c[available]-e[ix])**2).sum(1)); local=np.flatnonzero(available); j=local[np.argmin(ds)]
        if MATCH_CALIPER is not None and ds.min()>MATCH_CALIPER: continue
        available[j]=False;chosen.append((ix,j));distances.append(ds.min())
    chosen=sorted(chosen); return endpoint[[x[0] for x in chosen]],good[[x[1] for x in chosen]],np.asarray(distances)
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True);rng=np.random.default_rng(RANDOM_SEED);adj=pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_adjacent_pairs.csv')).query('both_included == 1');neigh={}
    for x in adj.itertuples(index=False):neigh.setdefault(x.roi_id_a,set()).add(x.roi_id_b);neigh.setdefault(x.roi_id_b,set()).add(x.roi_id_a)
    subjects=SUBJECT_LIST or sorted(os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(INPUT_DIR,'*_aim2_endpoint_covariates.npz')))
    for subj in subjects:
        z=np.load(os.path.join(INPUT_DIR,f'{subj}_aim2_endpoint_covariates.npz'),allow_pickle=True);cov=np.column_stack([z['wm_boundary_distance'],z['gm_probability'],z['tsnr']]);base=os.path.join(BATCH_DIR,subj);b=nib.load(os.path.join(base,'gm','func','derived',f'bnbmdenoised_detrended_rest_{subj}.nii'));d=b.get_fdata();gm=d[...,0]>0 if d.ndim==4 else d>0;n=int(gm.sum());lab=nib.load(os.path.join(base,'gm','derived',f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii')).get_fdata().astype(int)[gm];regions={r:np.flatnonzero(lab==r) for r in np.unique(lab) if r>0};R=np.load(os.path.join(base,'gm','func','derived','fc_results',f'{subj}_R.npz'))['arr_0'];W=np.load(os.path.join(base,'gm','anat','derived','dist_results',f'{subj}_W_a.npz'))['arr_0'];ii,jj=np.triu_indices(n,1);R=R[ii,jj] if R.ndim==2 else R;W=W[ii,jj] if W.ndim==2 else W;rows=[];balance=[]
        for k,tract in enumerate(z['tract']):
            src,remote=int(z['source_roi'][k]),int(z['remote_roi'][k]);e,c,md=match(z['endpoint_indices'][k],z['control_candidates'][k],cov,rng)
            if remote not in regions or len(e)<3: continue
            balance.append({'subject':subj,'tract':tract,'endpoint':int(z['endpoint'][k]),'n_matched':len(e),'mean_match_distance':np.mean(md),'smd_wm_distance':smd(cov[e,0],cov[c,0]),'smd_gm_probability':smd(cov[e,1],cov[c,1]),'smd_tsnr':smd(cov[e,2],cov[c,2])})
            targets=[('wm_connected_distant',remote)]+[('physically_adjacent',r) for r in neigh.get(src,set())-{remote} if r in regions]
            all_other=np.concatenate([x for roi,x in regions.items() if roi!=src])
            for cls,vox in [('endpoint',e),('matched_non_endpoint',c)]:
                slope=distance_slope(vox,all_other,n,R,W)
                for kind,target in targets:
                    f,distance,npairs=fc_summary(vox,regions[target],n,R,W);rows.append({'subject':subj,'tract_name':tract,'endpoint':int(z['endpoint'][k]),'source_roi':src,'target_roi':target,'target_type':kind,'voxel_class':cls,'mean_fc_z':f,'mean_distance_mm':distance,'n_matched_voxels':len(e),'n_voxel_pairs':npairs,'fc_distance_log_slope':slope,'wm_boundary_distance':np.mean(cov[vox,0]),'gm_probability':np.mean(cov[vox,1]),'tsnr':np.mean(cov[vox,2])})
        pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,f'{subj}_aim2_matched_endpoint_fc.csv'),index=False);pd.DataFrame(balance).to_csv(os.path.join(OUTPUT_DIR,f'{subj}_aim2_matching_balance.csv'),index=False)
if __name__=='__main__':main()
