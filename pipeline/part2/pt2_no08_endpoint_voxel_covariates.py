#!/usr/bin/env python3
# =============================================================================
# pt2_no8_endpoint_voxel_covariates.py
#
# AIM 2 (CONFOUNDER-CONTROLLED) / PURPOSE:
#   Define TRACULA endpoint GM voxels and save voxelwise confounders required
#   for matched non-endpoint controls: WM-boundary distance/cortical depth,
#   GM probability (SPM c1).  All images must share the BnB grid.
# INPUT: BnB mask, AAL image, aal_summary.csv, endpoint images, and 2 maps.
# OUTPUT: {subj}_aim2_endpoint_covariates.npz; aim2_no8_qc.csv.
# USER SETTINGS: edit only the section immediately below.
# =============================================================================
import os, glob
import numpy as np
import pandas as pd
import nibabel as nib

# =============================================================================
# USER SETTINGS
# =============================================================================
BATCH_DIR = '/path/to/your/project/derivatives/batch'
TEMPLATE_DIR = '/path/to/your/project/derivatives/template/AAL3'
OUTPUT_DIR = '/path/to/your/project/derivatives/pt2_group_results/no8_endpoint_covariates'
TRACT_SUFFIX = '_avg16_syn_bbr'
QC_EXCLUDE = {'rh.atr', 'acomm'}
SOURCE_SESSION = 'ses-01'
DARTEL_TEMPLATE_ID = 'YOUR_DARTEL_TEMPLATE'
# TRACULA fMRI-space endpoint masks.  Do not use endpt{end}.pd.nii here:
# ``fmri_endpt`` is the output intended to match the functional-space BnB/AAL.
ENDPOINT_GLOB_TEMPLATE = '{batch_dir}/{subject}/wm/freesurfer/{subject}/dpath/{tract}_avg16_syn_bbr/fmri_endpt{end}.pd.nii'
# Supply maps in subject functional/BnB space.  WM_DISTANCE may instead be a
# cortical-depth map, but it must have the same direction for every subject.
WM_DISTANCE_TEMPLATE = '{batch_dir}/{subject}/gm/anat/derived/aim2_wm_boundary_distance/wm_boundary_distance_c2thr0p5_{subject}.nii.gz'
GM_PROB_TEMPLATE = '{batch_dir}/{subject}/gm/func/preprocessing/c1mean{subject}_rest.nii'
# tSNR is intentionally not required.  It may be added later as a sensitivity
# covariate only if calculated from a non-detrended functional time series.
# Primary endpoint definition: retain voxels above 5% of that endpoint map's
# positive maximum after resampling to the fMRI/BnB grid.  The primary value is
# 0.05.  Do not change it for a sensitivity analysis unless the primary output
# has first been preserved deliberately.
ENDPOINT_RELATIVE_THRESHOLD = 0.05
SUBJECT_LIST = []
# =============================================================================

def path(template, **kwargs): return template.format(batch_dir=BATCH_DIR, **kwargs)
def image_values(file, reference, gm):
    img=nib.load(file)
    if img.shape[:3]!=reference.shape[:3] or not np.allclose(img.affine,reference.affine,atol=1e-3):
        raise ValueError('not aligned to BnB grid')
    return img.get_fdata()[gm]

def threshold_endpoint(file, reference, gm, relative_threshold):
    """Return fMRI-grid endpoint values above an auditable relative PD cutoff."""
    if not 0 < relative_threshold <= 1:
        raise ValueError('ENDPOINT_RELATIVE_THRESHOLD must be in (0, 1].')
    img = nib.load(file)
    if img.shape[:3] != reference.shape[:3] or not np.allclose(img.affine, reference.affine, atol=1e-3):
        raise ValueError('endpoint image is not aligned to BnB grid')
    data = img.get_fdata()
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        raise ValueError('endpoint image has no positive path-density values')
    pd_max = float(positive.max())
    cutoff = relative_threshold * pd_max
    return (np.isfinite(data[gm]) & (data[gm] > cutoff), pd_max, cutoff)

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True); include=set(pd.read_csv(os.path.join(TEMPLATE_DIR,'aal3_region_category.csv')).query('include == 1').roi_id)
    subjects=SUBJECT_LIST or sorted(s for s in os.listdir(BATCH_DIR) if s.isdigit() and os.path.isdir(os.path.join(BATCH_DIR,s))); qc=[]
    for subj in subjects:
        base=os.path.join(BATCH_DIR,subj); bfile=os.path.join(base,'gm','func','derived',f'bnbmdenoised_detrended_rest_{subj}.nii'); afile=os.path.join(base,'gm','derived',f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii'); sfile=os.path.join(base,'wm','derived','aal_summary',f'{subj}_aal_summary.csv')
        maps={'wm_boundary_distance':path(WM_DISTANCE_TEMPLATE,subject=subj),'gm_probability':path(GM_PROB_TEMPLATE,subject=subj)}
        if not all(os.path.exists(x) for x in [bfile,afile,sfile,*maps.values()]): qc.append({'subject':subj,'status':'skip','message':'missing core input or confounder map'}); continue
        try:
            b=nib.load(bfile); raw=b.get_fdata(); gm=raw[...,0]>0 if raw.ndim==4 else raw>0; labels=nib.load(afile).get_fdata().astype(int)[gm]; cov={name:image_values(f,b,gm) for name,f in maps.items()}
        except Exception as e: qc.append({'subject':subj,'status':'skip','message':str(e)}); continue
        region={r:np.flatnonzero(labels==r) for r in include if np.any(labels==r)}; tab=pd.read_csv(sfile); tab['tract_clean']=tab.tract.str.replace(TRACT_SUFFIX,'',regex=False); tab=tab[~tab.tract_clean.isin(QC_EXCLUDE)]; rec=[]
        for tract,g in tab.groupby('tract_clean'):
            rois={}
            for end in (1,2):
                x=g[g.endpoint==f'endpt{end}']; rois[end]=int(x.iloc[0].top1_aal_id) if not x.empty and pd.notna(x.iloc[0].top1_aal_id) else None
            if None in rois.values() or rois[1]==rois[2]: continue
            for end in (1,2):
                source,remote=rois[end],rois[3-end]; files=sorted(glob.glob(path(ENDPOINT_GLOB_TEMPLATE,subject=subj,tract=tract,end=end)))
                if source not in region or len(files)!=1: qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip','message':f'endpoint image matches={len(files)} or source ROI unavailable'}); continue
                try: ep, pd_max, cutoff = threshold_endpoint(files[0], b, gm, ENDPOINT_RELATIVE_THRESHOLD)
                except Exception as e: qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip','message':str(e)}); continue
                idx=region[source][ep[region[source]]]
                if len(idx):
                    rec.append((tract,end,source,remote,idx,pd_max,cutoff))
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'ok',
                               'message':f'endpoint voxels={len(idx)}; relative threshold={ENDPOINT_RELATIVE_THRESHOLD:g}; '
                                         f'absolute PD cutoff={cutoff:g}; map max={pd_max:g}'})
                else:
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip',
                               'message':f'no GM endpoint voxels at relative threshold={ENDPOINT_RELATIVE_THRESHOLD:g}; '
                                         f'absolute PD cutoff={cutoff:g}; map max={pd_max:g}'})
        union={r:set() for r in region}
        for _,_,source,_,idx,_,_ in rec: union[source].update(idx.tolist())
        out={k:[] for k in ['tract','endpoint','source_roi','remote_roi','endpoint_indices','control_candidates','endpoint_pd_max','endpoint_pd_cutoff']}
        for tract,end,source,remote,idx,pd_max,cutoff in rec:
            out['tract'].append(tract);out['endpoint'].append(end);out['source_roi'].append(source);out['remote_roi'].append(remote);out['endpoint_indices'].append(idx);out['control_candidates'].append(np.array(sorted(set(region[source])-union[source]),dtype=np.int64));out['endpoint_pd_max'].append(pd_max);out['endpoint_pd_cutoff'].append(cutoff)
        out['endpoint_relative_threshold'] = np.array([ENDPOINT_RELATIVE_THRESHOLD], dtype=float)
        out.update({k:np.asarray(v) for k,v in cov.items()}); np.savez_compressed(os.path.join(OUTPUT_DIR,f'{subj}_aim2_endpoint_covariates.npz'),**{k:np.asarray(v,dtype=object) if k in ['tract','endpoint','source_roi','remote_roi','endpoint_indices','control_candidates'] else v for k,v in out.items()})
    pd.DataFrame(qc).to_csv(os.path.join(OUTPUT_DIR,'aim2_no8_qc.csv'),index=False)
if __name__=='__main__': main()
