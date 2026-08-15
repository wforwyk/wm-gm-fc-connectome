#!/usr/bin/env python3
# =============================================================================
# pt2_no7_wm_boundary_distance.py
#
# AIM 2 / PURPOSE:
#   Create a subject-specific WM-boundary-distance map in the BnB functional
#   grid.  The WM mask comes from SPM c2mean{subj}_rest.nii.  The map stores,
#   in mm, each voxel's Euclidean distance to the nearest WM voxel.
#
# INPUT (per subject): BnB functional reference; c2mean WM-probability image.
# OUTPUT: wm_boundary_distance_c2thr{threshold}_{subj}.nii.gz, plus the binary
#   WM mask used for each threshold.  Thresholds support sensitivity analyses.
# USER SETTINGS: edit only the block below.
# =============================================================================
import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt

# =============================================================================
# USER SETTINGS
# =============================================================================
BATCH_DIR = '/path/to/your/project/derivatives/batch'
OUTPUT_SUBDIR = os.path.join('gm', 'anat', 'derived', 'aim2_wm_boundary_distance')
# c2 is the SPM WM probability map made during functional preprocessing.
C2_TEMPLATE = '{batch_dir}/{subject}/gm/func/preprocessing/c2mean{subject}_rest.nii'
BNB_TEMPLATE = '{batch_dir}/{subject}/gm/func/derived/bnbmdenoised_detrended_rest_{subject}.nii'
# Primary uses 0.5.  0.7 and 0.9 are sensitivity analyses, not additional
# input requirements for the primary Aim 2 model.
WM_PROB_THRESHOLDS = [0.5, 0.7, 0.9]
SUBJECT_LIST = []
# =============================================================================

def main():
    subjects = SUBJECT_LIST or sorted(s for s in os.listdir(BATCH_DIR)
                                      if s.isdigit() and os.path.isdir(os.path.join(BATCH_DIR, s)))
    for subject in subjects:
        bnb_path = BNB_TEMPLATE.format(batch_dir=BATCH_DIR, subject=subject)
        c2_path = C2_TEMPLATE.format(batch_dir=BATCH_DIR, subject=subject)
        if not os.path.exists(bnb_path) or not os.path.exists(c2_path):
            print(f'{subject}: SKIP missing BnB or c2mean'); continue
        reference = nib.load(bnb_path)
        c2 = nib.load(c2_path)
        target = (reference.shape[:3], reference.affine)
        # Probability images use linear interpolation if resampling is needed.
        if c2.shape[:3] != reference.shape[:3] or not np.allclose(c2.affine, reference.affine, atol=1e-3):
            c2 = resample_from_to(c2, target, order=1)
            print(f'{subject}: c2 resampled to BnB grid')
        c2_data = c2.get_fdata()
        voxel_mm = np.sqrt((reference.affine[:3, :3] ** 2).sum(axis=0))
        outdir = os.path.join(BATCH_DIR, subject, OUTPUT_SUBDIR)
        os.makedirs(outdir, exist_ok=True)
        for threshold in WM_PROB_THRESHOLDS:
            wm = np.isfinite(c2_data) & (c2_data >= threshold)
            if not wm.any():
                print(f'{subject}: threshold {threshold}: SKIP no WM voxels'); continue
            # EDT measures every non-WM voxel's distance to the nearest zero
            # (WM) voxel.  Values are physical mm using the BnB voxel spacing.
            distance = distance_transform_edt(~wm, sampling=voxel_mm).astype(np.float32)
            tag = str(threshold).replace('.', 'p')
            nib.save(nib.Nifti1Image(wm.astype(np.uint8), reference.affine, reference.header),
                     os.path.join(outdir, f'wm_mask_c2thr{tag}_{subject}.nii.gz'))
            nib.save(nib.Nifti1Image(distance, reference.affine, reference.header),
                     os.path.join(outdir, f'wm_boundary_distance_c2thr{tag}_{subject}.nii.gz'))
        print(f'{subject}: WM-boundary distance maps written')

if __name__ == '__main__':
    main()
