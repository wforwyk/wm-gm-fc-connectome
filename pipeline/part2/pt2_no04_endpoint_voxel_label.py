#!/usr/bin/env python3
# =============================================================================
# pt2_no4_endpoint_voxel_label.py
#
# AIM 2 / PURPOSE:
#   Label GM voxels at each QC-passing TRACULA tract endpoint.  Endpoint voxels
#   are the intersection of the tract endpoint image, BnB GM mask, and the AAL
#   region assigned in aal_summary.csv.  Non-endpoint voxels in that same AAL
#   region are saved as matched-control candidates for pt2_no5.
#
# INPUT (per subject): BnB mask; warped AAL image; aal_summary.csv; TRACULA
#   endpoint NIfTI images.  Endpoint images MUST be in the BnB/AAL voxel grid.
# OUTPUT: {subj}_endpoint_voxel_labels.npz and endpoint_voxel_label_qc.csv
# USER SETTINGS: edit only the block below, especially ENDPOINT_GLOB_TEMPLATE.
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
OUTPUT_DIR = '/path/to/your/project/derivatives/pt2_group_results/no4_endpoint_voxels'
TRACT_SUFFIX = '_avg16_syn_bbr'
QC_EXCLUDE = {'rh.atr', 'acomm'}
SOURCE_SESSION = 'ses-01'
DARTEL_TEMPLATE_ID = 'YOUR_DARTEL_TEMPLATE'
# Produced by part1/template/pt1_template_no05_coregister_endpoints_to_fmri.m.
# It must resolve to exactly one image. Available fields: {batch_dir},
# {subject}, {tract}, {end} (end is 1 or 2). ``tract`` is the cleaned name
# from aal_summary (e.g., cc.bodypf); its directory retains TRACT_SUFFIX.
ENDPOINT_GLOB_TEMPLATE = '{batch_dir}/{subject}/wm/freesurfer/{subject}/dpath/{tract}_avg16_syn_bbr/fmri_endpt{end}.pd.nii'
# Primary endpoint definition: retain voxels above 5% of that endpoint map's
# positive maximum after resampling to the fMRI/BnB grid.  This removes the
# low-amplitude interpolation tail while adapting to each tract-end's PD scale.
# The prespecified sensitivity values are 0.01 and 0.10.  Change this setting
# only after preserving the primary output deliberately.
ENDPOINT_RELATIVE_THRESHOLD = 0.05
# Leave empty to process every numeric subject directory.
SUBJECT_LIST = []
# =============================================================================

def endpoint_file(subject, tract, end):
    pattern = ENDPOINT_GLOB_TEMPLATE.format(batch_dir=BATCH_DIR, subject=subject, tract=tract, end=end)
    files = sorted(glob.glob(pattern))
    return files[0] if len(files) == 1 else None, pattern, len(files)

def threshold_endpoint(data, relative_threshold):
    """Return a positive-PD endpoint mask and its auditable threshold values."""
    if not 0 < relative_threshold <= 1:
        raise ValueError('ENDPOINT_RELATIVE_THRESHOLD must be in (0, 1].')
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        raise ValueError('endpoint image has no positive path-density values')
    pd_max = float(positive.max())
    cutoff = relative_threshold * pd_max
    return np.isfinite(data) & (data > cutoff), pd_max, cutoff

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    included = set(pd.read_csv(os.path.join(TEMPLATE_DIR, 'aal3_region_category.csv')).query('include == 1').roi_id)
    subjects = SUBJECT_LIST or sorted(s for s in os.listdir(BATCH_DIR) if s.isdigit() and os.path.isdir(os.path.join(BATCH_DIR, s)))
    qc = []
    for subj in subjects:
        base = os.path.join(BATCH_DIR, subj)
        bnb_path = os.path.join(base, 'gm', 'func', 'derived', f'bnbmdenoised_detrended_rest_{subj}.nii')
        aal_path = os.path.join(base, 'gm', 'derived', f'fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii')
        summary_path = os.path.join(base, 'wm', 'derived', 'aal_summary', f'{subj}_aal_summary.csv')
        if not all(os.path.exists(x) for x in [bnb_path, aal_path, summary_path]):
            qc.append({'subject': subj, 'status': 'skip', 'message': 'missing BnB, AAL, or aal_summary input'}); continue
        bnb, aal = nib.load(bnb_path), nib.load(aal_path)
        d = bnb.get_fdata(); gm = d[..., 0] > 0 if d.ndim == 4 else d > 0
        labels = aal.get_fdata().astype(int)[gm]
        region_idx = {r: np.flatnonzero(labels == r) for r in included if np.any(labels == r)}
        tab = pd.read_csv(summary_path)
        tab['tract_clean'] = tab['tract'].str.replace(TRACT_SUFFIX, '', regex=False)
        tab = tab[~tab.tract_clean.isin(QC_EXCLUDE)]
        records = []
        for tract, g in tab.groupby('tract_clean'):
            roi = {}
            for end in (1, 2):
                x = g[g.endpoint == f'endpt{end}']
                if not x.empty and pd.notna(x.iloc[0].top1_aal_id): roi[end] = int(x.iloc[0].top1_aal_id)
            if len(roi) != 2 or roi[1] == roi[2]: continue
            for end in (1, 2):
                source, remote = roi[end], roi[2 if end == 1 else 1]
                path, pattern, nfiles = endpoint_file(subj, tract, end)
                if source not in region_idx or path is None:
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip','message':f'ROI unavailable or endpoint glob found {nfiles}: {pattern}'}); continue
                eimg = nib.load(path)
                if eimg.shape[:3] != gm.shape or not np.allclose(eimg.affine, bnb.affine, atol=1e-3):
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip','message':'endpoint image is not in BnB grid'}); continue
                try:
                    endpoint_mask, pd_max, cutoff = threshold_endpoint(
                        eimg.get_fdata(), ENDPOINT_RELATIVE_THRESHOLD
                    )
                except ValueError as exc:
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip','message':str(exc)})
                    continue
                endpoint = region_idx[source][endpoint_mask[gm][region_idx[source]]]
                if len(endpoint) == 0:
                    qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'skip',
                               'message':f'no GM endpoint voxels at relative threshold {ENDPOINT_RELATIVE_THRESHOLD:g} '
                                         f'(absolute PD cutoff {cutoff:g}; map max {pd_max:g})'})
                    continue
                records.append((tract, end, source, remote, endpoint))
                qc.append({'subject':subj,'tract':tract,'endpoint':end,'status':'ok',
                           'message':f'{len(endpoint)} endpoint voxels; relative threshold '
                                     f'{ENDPOINT_RELATIVE_THRESHOLD:g}; absolute PD cutoff {cutoff:g}; map max {pd_max:g}'})
        # Union across tracts prevents an endpoint of another tract being a "control".
        union = {r:set() for r in region_idx}
        for _,_,source,_,idx in records: union[source].update(idx.tolist())
        out = {'tract':[], 'endpoint':[], 'source_roi':[], 'remote_roi':[], 'endpoint_indices':[], 'control_candidates':[]}
        for tract,end,source,remote,idx in records:
            controls = np.array(sorted(set(region_idx[source]) - union[source]), dtype=np.int64)
            out['tract'].append(tract); out['endpoint'].append(end); out['source_roi'].append(source); out['remote_roi'].append(remote)
            out['endpoint_indices'].append(idx); out['control_candidates'].append(controls)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f'{subj}_endpoint_voxel_labels.npz'),
                            **{k:np.array(v, dtype=object) for k,v in out.items()})
        print(f'{subj}: {len(records)} usable tract endpoints')
    pd.DataFrame(qc).to_csv(os.path.join(OUTPUT_DIR, 'endpoint_voxel_label_qc.csv'), index=False)

if __name__ == '__main__': main()
