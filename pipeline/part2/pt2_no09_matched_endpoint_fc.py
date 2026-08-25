#!/usr/bin/env python3
# =============================================================================
# pt2_no9_matched_endpoint_fc.py
#
# AIM 2 (CONFOUNDER-CONTROLLED) / PURPOSE:
#   Match every endpoint voxel to one non-endpoint voxel in the same AAL region
#   using WM-boundary distance and GM probability.  Compare matched sets'
#   FC to WM-connected distant versus physically adjacent AAL regions.
# INPUT: pt2_no8 NPZ; Aim 1 BnB/AAL images; R and W matrices; adjacency CSV.
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
BATCH_DIR='/path/to/your/project/derivatives/batch'
TEMPLATE_DIR='/path/to/your/project/derivatives/template/AAL3'
INPUT_DIR='/path/to/your/project/derivatives/pt2_group_results/no8_endpoint_covariates'
OUTPUT_DIR='/path/to/your/project/derivatives/pt2_group_results/no9_matched_endpoint_fc'
MATCH_CALIPER=2.0  # maximum standardized Euclidean distance; set None to disable
RANDOM_SEED=20260729
MIN_MATCHED_VOXELS=10  # primary matched sensitivity-analysis inclusion criterion
SUBJECT_LIST=[]
SOURCE_SESSION = 'ses-01'
DARTEL_TEMPLATE_ID = 'YOUR_DARTEL_TEMPLATE'
# =============================================================================

MATCHED_FC_COLUMNS=['subject','tract_name','endpoint','source_roi','target_roi','target_type','voxel_class','endpoint_relative_threshold','mean_fc_z','mean_distance_mm','n_matched_voxels','n_voxel_pairs','fc_distance_log_slope','wm_boundary_distance','gm_probability']
BALANCE_COLUMNS=['subject','tract','endpoint','endpoint_relative_threshold','n_matched','mean_match_distance','smd_wm_distance','smd_gm_probability']

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
    # A small or unavailable candidate pool simply yields a partial match.  It
    # is then excluded transparently by MIN_MATCHED_VOXELS, never averaged empty.
    if len(endpoint)==0 or len(good)==0: return np.array([],dtype=int),np.array([],dtype=int),np.array([])
    allx=np.vstack([cov[endpoint],cov[good]]); mu=allx.mean(0); sd=allx.std(0); sd[sd==0]=1; e=(cov[endpoint]-mu)/sd; c=(cov[good]-mu)/sd; available=np.ones(len(good),bool); chosen=[]; distances=[]
    for ix in rng.permutation(len(endpoint)):
        # Controls are used without replacement.  Once the pool is exhausted,
        # retain the valid partial match rather than calling argmin on an empty
        # candidate set.
        if not available.any(): break
        ds=np.sqrt(((c[available]-e[ix])**2).sum(1)); local=np.flatnonzero(available); j=local[np.argmin(ds)]
        if MATCH_CALIPER is not None and ds.min()>MATCH_CALIPER: continue
        available[j]=False;chosen.append((ix,j));distances.append(ds.min())
    chosen=sorted(chosen); return endpoint[[x[0] for x in chosen]],good[[x[1] for x in chosen]],np.asarray(distances)
def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True);rng=np.random.default_rng(RANDOM_SEED);adj=pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_adjacent_pairs.csv')).query('both_included == 1');neigh={}
    for x in adj.itertuples(index=False):neigh.setdefault(x.roi_id_a,set()).add(x.roi_id_b);neigh.setdefault(x.roi_id_b,set()).add(x.roi_id_a)
    subjects=SUBJECT_LIST or sorted(os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(INPUT_DIR,'*_aim2_endpoint_covariates.npz')))
    for subj in subjects:
        z=np.load(os.path.join(INPUT_DIR,f'{subj}_aim2_endpoint_covariates.npz'),allow_pickle=True)
        if 'endpoint_relative_threshold' not in z:
            raise RuntimeError(f'{subj}: legacy no08 input lacks endpoint_relative_threshold; rerun pt2_no08.')
        endpoint_relative_threshold=float(np.asarray(z['endpoint_relative_threshold']).ravel()[0])
        cov=np.column_stack([z['wm_boundary_distance'],z['gm_probability']]);base=os.path.join(BATCH_DIR,subj);b=nib.load(os.path.join(base,'gm','func','derived',f'bnbmdenoised_detrended_rest_{subj}.nii'));d=b.get_fdata();gm=d[...,0]>0 if d.ndim==4 else d>0;n=int(gm.sum());lab=nib.load(os.path.join(base,'gm','derived',f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii')).get_fdata().astype(int)[gm];regions={r:np.flatnonzero(lab==r) for r in np.unique(lab) if r>0};R=np.load(os.path.join(base,'gm','func','derived','fc_results',f'{subj}_R.npz'))['arr_0'];W=np.load(os.path.join(base,'gm','anat','derived','dist_results',f'{subj}_W_a.npz'))['arr_0'];ii,jj=np.triu_indices(n,1);R=R[ii,jj] if R.ndim==2 else R;W=W[ii,jj] if W.ndim==2 else W;rows=[];balance=[]
        for k,tract in enumerate(z['tract']):
            src,remote=int(z['source_roi'][k]),int(z['remote_roi'][k]);e,c,md=match(z['endpoint_indices'][k],z['control_candidates'][k],cov,rng)
            balance.append({'subject':subj,'tract':tract,'endpoint':int(z['endpoint'][k]),'endpoint_relative_threshold':endpoint_relative_threshold,'n_matched':len(e),'mean_match_distance':np.mean(md) if len(md) else np.nan,'smd_wm_distance':smd(cov[e,0],cov[c,0]) if len(e)>1 else np.nan,'smd_gm_probability':smd(cov[e,1],cov[c,1]) if len(e)>1 else np.nan})
            if remote not in regions or len(e)<MIN_MATCHED_VOXELS: continue
            targets=[('wm_connected_distant',remote)]+[('physically_adjacent',r) for r in neigh.get(src,set())-{remote} if r in regions]
            all_other=np.concatenate([x for roi,x in regions.items() if roi!=src])
            for cls,vox in [('endpoint',e),('matched_non_endpoint',c)]:
                slope=distance_slope(vox,all_other,n,R,W)
                for kind,target in targets:
                    f,distance,npairs=fc_summary(vox,regions[target],n,R,W);rows.append({'subject':subj,'tract_name':tract,'endpoint':int(z['endpoint'][k]),'source_roi':src,'target_roi':target,'target_type':kind,'voxel_class':cls,'endpoint_relative_threshold':endpoint_relative_threshold,'mean_fc_z':f,'mean_distance_mm':distance,'n_matched_voxels':len(e),'n_voxel_pairs':npairs,'fc_distance_log_slope':slope,'wm_boundary_distance':np.mean(cov[vox,0]),'gm_probability':np.mean(cov[vox,1])})
        # Preserve headers even for subjects without eligible matched units so
        # downstream code can identify and skip them without a CSV parse error.
        pd.DataFrame(rows,columns=MATCHED_FC_COLUMNS).to_csv(os.path.join(OUTPUT_DIR,f'{subj}_aim2_matched_endpoint_fc.csv'),index=False)
        pd.DataFrame(balance,columns=BALANCE_COLUMNS).to_csv(os.path.join(OUTPUT_DIR,f'{subj}_aim2_matching_balance.csv'),index=False)
if __name__=='__main__':main()
