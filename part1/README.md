# BrainWorld WM–GM-FC Pipeline

This README is written based on example subjects 1001 and 1002, with subject list group 'NCKU_336Ss'. Provided scripts and manuals may include subject names, path directories, and file names that must be changed for further use. All scripts have a USER SETTINGS block at the top — check and update this before running each script.

## Purpose and Scope

This manual describes the full analysis pipeline for the BrainWorld project, which integrates grey matter (GM) functional connectivity and voxel-wise distance metrics with white matter (WM) tract-based structural information in individual subject space.

The pipeline operates strictly at the subject level (no group-space analysis in the main workflow).

## High-Level Pipeline Overview

The pipeline is organised into three major blocks:

1. GM Pipeline — Preprocessing, denoising, FC matrix, and geodesic distance matrix
2. Template Preparation — Warp AAL atlas from MNI -> SST -> subject space (run once before WM)
3. WM Pipeline — TRACULA tractography, endpoint/pathstats extraction, AAL integration, WM-GM-FC combined outputs

The GM pipeline must be completed before running the WM pipeline, as GM outputs (c1, R.npz, W_a.npz, AAL atlas files in gm/derived/) are required by multiple WM scripts.

## Software Requirements

Software        | Notes
----------------|---------------------------------------------------------------
MATLAB R2025b   | with SPM12 on path (/opt/spm)
SPM12           | TPM at /opt/spm/tpm/TPM.nii
FreeSurfer>=7.2 | <7.2.0 only supports 18 TRACULA tracts; 42 tracts require >=7.2.0
FSL             | fslstats, fslinfo used in no4 and template verification
ANTs            | Required by no3_trac-all.sh; set ANTSPATH
DSI Studio      | Used in Template Preparation (manual step)
Python 3        | nibabel, numpy, scipy, pandas, matplotlib, plotly (optional)

## Directory Structure (per subject)

```
{BASE_DIR}/{subj}/
├── gm/
│   ├── anat/
│   │   ├── segmentation/          <- T1, c1*.nii (GM segmentation)
│   │   └── derived/dist_results/  <- A.npz, W_a.npz, vector_r/w.npy, path/node_*.npz
│   ├── derived/                   <- u_rc1*.nii, wsst/fmri_/dwi_ AAL atlas files
│   └── func/
│       ├── preprocessing/         <- rest.nii and SPM preprocessing outputs
│       ├── model/1st_level/       <- SPM.mat, beta_*.nii, residuals/Res_*.nii
│       └── derived/
│           ├── fc_results/        <- R.npz
│           ├── denoised_detrended_rest_{subj}.nii
│           ├── mdenoised_detrended_rest_{subj}.nii
│           └── bmdenoised_detrended_rest_{subj}.nii
├── wm/
│   ├── freesurfer/{subj}/         <- recon-all + TRACULA outputs (dpath/, dmri/, etc.)
│   ├── tracula/{subj}_dmrirc      <- TRACULA config file
│   └── derived/
│       ├── endpoints/             <- endpt1/, endpt2/ CSVs + endpoints_size.csv
│       ├── pathstats/             <- {subj}_pathstats.csv
│       └── aal_summary/
│           ├── {subj}_aal_summary.csv
│           ├── heatmap/           <- AAL-AAL matrices + edge details + PNG
│           ├── rank_edge/         <- edge rankings per metric
│           └── rank_roi/          <- ROI rankings per metric
└── wm-gm-fc/
    ├── 3d/                        <- voxelpair CSVs + 3D scatter PNG/HTML
    └── heatmaps/                  <- heatmap_A_fc, B_dist, C_rd PNGs
```

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
---------------------------------- G M --------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

## GM Pipeline Description

The GM pipeline consists of two tightly coupled parts:
- MATLAB-based processing (no1_2 ~ no5): preprocessing, denoising via GLM, segmentation, masking
- Python-based analysis (no6 ~ no7): voxel-wise FC, geodesic distance, and index vectors

All GM steps are executed per subject and must be completed before running the WM pipeline,
as GM segmentation output (c1) and atlas files are reused by WM scripts.

### Execution Order

```
0. no0_setting_directory_data.sh      (bash)
1~5. no1_2_preprocessing.m            (MATLAB / SPM)
     no3_denoise.m
     no4_modelestimation_detrend.m
     no5_masking_binarise.m
6~7. no6_corr_mat_dist.py             (Python)
     no7_indices_and_output.py
```

---

### Step 0: no0_setting_directory_data.sh

Sets up the full directory structure and copies source data into the project.

USER SETTINGS: BASE_DIR, SRC_DIR, SES_SRC, SES_DST, SUBJECTS

What it does:
1. Copies T1 and resting-state from SRC_DIR to rawdata/
   (`{subj}_{SES_SRC}_T1w.nii` -> `rawdata/sub-{subj}/{SES_DST}/anat/{subj}_t1.nii`)
2. Creates the full `derivatives/{subj}/gm/` and `derivatives/{subj}/wm/` directory tree
3. Copies T1 to `gm/anat/segmentation/` and resting-state to `gm/func/preprocessing/`

Note: `rawdata/` uses `sub-XXXX` prefix; `derivatives/` uses numeric ID only (e.g., 1001).

Note: this script pre-creates `gm/func/derived/fc_results` and `gm/func/derived/distance/path`, not `gm/anat/derived/dist_results/path` (the path actually used by no6/no7). This is harmless — no6 creates `gm/anat/derived/dist_results/` itself via `os.makedirs(..., exist_ok=True)` — but it does leave an unused `gm/func/derived/distance/` folder. Worth a look if you want no0 to match the real output tree exactly.

---

### Steps 1-2: no1_2_preprocessing.m

SPM fMRI preprocessing pipeline. Runs as a subject loop.

USER SETTINGS: BASE_DIR, SUBJECTS, N_VOLS, SPM_ROOT, TPM_PATH, BIAS_FUN,
               and slice timing parameters (ST_NSLICES, ST_TR, ST_TA, ST_SO, ST_REFSLICE)

Input:
- `gm/func/preprocessing/{subj}_rest.nii`  (N_VOLS volumes)
- `gm/anat/segmentation/{subj}_t1.nii`

SPM jobs (run in sequence):
1. Realign: Estimate & Reslice
   -> `r{subj}_rest.nii`, `mean{subj}_rest.nii`, `rp_{subj}_rest.txt` (motion parameters)
2. Segment (mean func, for bias field extraction)
   -> `BiasField_mean{subj}_rest.nii`
3. Apply Bias Field (calls `apply_bias_field_image.m`)
   -> bias-corrected volumes (`b`-prefixed)
4. Slice Timing Correction
   -> `abr{subj}_rest.nii`
5. Coregister: Estimate (T1 -> mean func space)
6. Segment (coregistered T1)
   -> `c1{subj}_t1.nii` (GM probability map) and tissue classes c1-c5

Key notes:
- Confirm slice timing parameters match your data (fslinfo / fslhd)
- Ensure `apply_bias_field_image.m` is on the MATLAB path and BIAS_FUN points to it
- Pre-saved settings match NCKU data; change slice timing parameters if using other data
- `affreg = 'eastern'` is set for Asian brain templates

---

### Step 3: no3_denoise.m

GLM-based denoising via SPM 1st-level model specification and estimation.

USER SETTINGS: BASE_DIR, SUBJECTS, N_VOLS, SPM_ROOT

Input:
- `gm/func/preprocessing/abr{subj}_rest.nii`  (slice-timing corrected, 240 volumes)
- `gm/func/preprocessing/rp_{subj}_rest.txt`  (motion parameters as nuisance regressors)

Output: (`gm/func/model/1st_level/`)
- `SPM.mat`
- `beta_000*.nii`  (motion regressor betas)
- `Res_000*.nii`   (residuals = denoised signal; used by no4)

---

### Step 4: no4_modelestimation_detrend.m

Model estimation -> mask reslice -> detrend -> output final denoised/detrended time series.

USER SETTINGS: BASE_DIR, SUBJECTS, SPM_ROOT

Input:
- `gm/func/model/1st_level/SPM.mat`
- `gm/anat/segmentation/c1{subj}_t1.nii`

What it does:
1. Re-estimates GLM model; writes `Res_*.nii` residuals
2. Reslices c1 mask to match fMRI resolution (`rc1{subj}_t1.nii`)
3. Detrends the residual time series within the GM mask (threshold > 0.2)
4. Saves detrended output as `gm/func/derived/denoised_detrended_rest_{subj}.nii`
5. Moves `Res_*.nii` into `model/1st_level/residuals/`

Output:
- `gm/func/derived/denoised_detrended_rest_{subj}.nii`

---

### Step 5: no5_masking_binarise.m

Applies GM mask and binarises.

USER SETTINGS: BASE_DIR, SUBJECTS, SPM_ROOT

Input:
- `gm/anat/segmentation/c1{subj}_t1.nii`
- `gm/func/derived/denoised_detrended_rest_{subj}.nii`

What it does:
1. Masking (spm_mask, threshold 0.9):
   -> `mdenoised_detrended_rest_{subj}.nii`
2. Binarisation (values > 0):
   -> `bmdenoised_detrended_rest_{subj}.nii`

Output:
- `gm/func/derived/mdenoised_detrended_rest_{subj}.nii`
- `gm/func/derived/bmdenoised_detrended_rest_{subj}.nii`

---

### Step 6: no6_corr_mat_dist.py

Computes voxel-wise FC correlation matrix and GM geodesic distance matrix.

USER SETTINGS: SUBJECTS, BASE_DIR, NUM_PROCESSES (parallel workers for adjacency matrix)

Input:
- `gm/func/derived/denoised_detrended_rest_{subj}.nii`
- `gm/func/derived/bmdenoised_detrended_rest_{subj}.nii`  (GM mask)

What it does:
1. Loads functional data and builds GM voxel set (vset) from mask AND variance > 0
2. Saves updated GM mask: `bnbmdenoised_detrended_rest_{subj}.nii`
3. Computes voxel x voxel Pearson correlation matrix
4. Builds sparse adjacency matrix (26-neighbourhood, anisotropic voxel distances)
5. Computes Dijkstra shortest paths; saves per-node predecessors to `dist_results/path/node_*.npz`
6. Extracts GM-masked distance submatrix; removes temp `W.dat`

Output:
- `gm/func/derived/fc_results/{subj}_R.npz`           -- FC correlation matrix
- `gm/anat/derived/dist_results/{subj}_A.npz`         -- sparse adjacency matrix
- `gm/anat/derived/dist_results/{subj}_W_a.npz`       -- GM-masked geodesic distance matrix
- `gm/func/derived/bnbmdenoised_detrended_rest_{subj}.nii`  -- updated GM mask
- `gm/anat/derived/dist_results/path/node_*.npz`      -- Dijkstra predecessor arrays

Note: This step is computationally intensive (Dijkstra over all GM voxels).
      NUM_PROCESSES controls parallel adjacency computation.

---

### Step 7: no7_indices_and_output.py

Derives upper-triangle vectors from FC and distance matrices, and computes their correlation.

USER SETTINGS: SUBJECTS, BASE_DIR,
               INF_ROW_THRESHOLD (default 100; removes isolated voxel clusters)

Input:
- `gm/func/derived/fc_results/{subj}_R.npz`
- `gm/anat/derived/dist_results/{subj}_W_a.npz`

What it does:
1. Loads R and W_a; writes to memory-mapped files for row-wise access
2. Removes isolated voxel rows (rows with >= n - INF_ROW_THRESHOLD inf values in W)
3. Extracts upper-triangle (k=1) vectors from FC and distance matrices
4. Prints FC-distance Pearson correlation coefficient
5. Saves vectors; cleans up temp memmap files

Output:
- `gm/anat/derived/dist_results/vector_r_{subj}.npy`  -- upper-triangle FC values
- `gm/anat/derived/dist_results/vector_w_{subj}.npy`  -- upper-triangle geodesic distance values

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
---------------------------- TEMPLATE PREP ----------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

## Template Preparation: MNI -> SST -> Subject Space

This stage warps the AAL atlas into each subject's native space. It must be completed
before running the WM pipeline (no7 onward), as the resulting `dwi_wsst_...nii` and
`fmri_wsst_...nii` files are used by WM scripts.

The MNI -> SST step (tp_no1) is performed once and shared across all subjects.
The SST -> subject steps (tp_no2, tp_no3) are run per subject.

### Execution Order

```
tp_no0. no0_copy_u_rc1.sh             (bash)   -- copy DARTEL flow fields
tp_no1. [Manual: DSI Studio + SPM]    (manual) -- MNI -> SST (once)
tp_no2. no2_SST_to_sub.m              (MATLAB) -- SST -> subject T1 space
tp_no3. no3_coregister_to_dwi_fmri.m  (MATLAB) -- T1 -> fMRI space + DWI space
tp_no4. no4_verification.sh           (bash)   -- verify outputs
```

---

### tp_no0: no0_copy_u_rc1.sh

Copies the DARTEL flow field (`u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`)
from the source DARTEL directory into each subject's `gm/derived/` folder.

USER SETTINGS: SUBJECTS, SRC_DIR, TEMPLATE_DIR

Input:  `{SRC_DIR}/sub-{subj}/ses-1/anat/u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
Output: `{BASE_DIR}/{subj}/gm/derived/u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`

---

### tp_no1: DSI Studio + SPM (Manual, run once)

Purpose: Export AAL atlas in DSI Studio space and warp to SST.
         Output is shared across all subjects.

Steps:
1. Load NTU-DSI-122 in DSI Studio -> click 'Reconstruct'
2. Load `ROI_MNI_V7_1mm.nii` as 'Load regions'
   - Edit `ROI_MNI_V7_1mm.txt` to contain only ROI numbers and region names
     (DSI Studio requires this format to load region names correctly)
3. Export all ROI regions -> `all_ROIs_AAL.nii`
4. In MATLAB/SPM, load batch `NTU-DSI_to_SST.mat` [Deformation @ SPM]
   - Flow field: `iy_NCKU_336Ss_6.nii`
   - Input:      `all_ROIs_AAL.nii`
   - Output:     `sst_all_ROIs_AAL.nii`

Save `sst_all_ROIs_AAL.nii` and `iy_NCKU_336Ss_6.nii` together in the shared
template folder (e.g., `derivatives/template/NCKU/`).

---

### tp_no2: no2_SST_to_sub.m

Warps `sst_all_ROIs_AAL.nii` from SST space into each subject's native T1 space
using the DARTEL flow field.

USER SETTINGS: SUBJECTS, BASE_DIR, TEMPLATE_DIR (path to sst_all_ROIs_AAL.nii)

Input:
- `{BASE_DIR}/{subj}/gm/derived/u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `{TEMPLATE_DIR}/sst_all_ROIs_AAL.nii`

SPM batch: `spm.tools.dartel.crt_iwarped`  (K=6, interp=1 trilinear)

Output: `{BASE_DIR}/{subj}/gm/derived/wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`

---

### tp_no3: no3_coregister_to_dwi_fmri.m

Coregisters the T1-space AAL atlas into fMRI space and DWI space.
Uses nearest-neighbour interpolation (interp=0) to preserve integer AAL labels.

USER SETTINGS: SUBJECTS, BASE_DIR

Input:
- `gm/derived/wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `gm/func/preprocessing/mean{subj}_rest.nii`  (fMRI reference)
- `wm/freesurfer/{subj}/dmri/dwi.nii.gz`       (automatically extracted to dwi.nii)

What it does:
1. Gunzip `dwi.nii.gz` -> `dwi.nii` (skipped if already exists)
2. Coreg AAL T1 -> fMRI space  (ref: mean{subj}_rest.nii, prefix: fmri_)
3. Coreg AAL T1 -> DWI space   (ref: dwi.nii,             prefix: dwi_)

Output: (both saved to `gm/derived/`)
- `fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`

Note: interp=0 (nearest neighbour) is used here to preserve discrete AAL label integers,
      vs. interp=1 in tp_no2 which uses a continuous warp field.

---

### tp_no4: no4_verification.sh

Verifies that all three AAL atlas files (T1, fMRI, DWI space) have correct
shape, voxel size, and label range.

USER SETTINGS: SUBJECTS, BASE_DIR,
               AAL_EXPECTED_MIN / AAL_EXPECTED_MAX (default 100-170;
               AAL2=116, AAL3=166 -- both accepted)

Checks per subject:
1. All 3 files exist (wsst_, fmri_, dwi_)
2. fMRI-space AAL shape/voxel matches mean{subj}_rest.nii
3. DWI-space AAL shape/voxel matches dwi.nii
4. AAL label max in valid range (100-170)

Output: summary printed to terminal with a warnings list at the end.

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
---------------------------------- W M --------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

## WM Pipeline Description

WM tracts are reconstructed in individual subject space across 40 predefined pathways
(42 total minus lh.cst and rh.cst) using TRACULA. Tract endpoints are integrated with
the DWI-space AAL atlas to derive AAL-labelled structural connectivity measures, then
combined with GM FC and distance for final WM-GM-FC outputs.

### Execution Order

```
1.  no1_recon-all.sh                (bash)   -- FreeSurfer recon-all + thalamic nuclei
2.  [Manual] no2_{subj}_dmrirc      (manual) -- write TRACULA config per subject
3.  no3_trac-all.sh                 (bash)   -- TRACULA (-prep, -bedp, -path)
4.  no4_endpoints_coordinates.py    (Python)
5.  no5_endpoints_size.py           (Python)
6.  no6_pathstats.py                (Python)
7.  no7_aal_summary.py              (Python)
8.  no8_aal_heatmap.py              (Python)
9.  no9_aal_rank_edge.py            (Python)
10. no10_aal_rank_roi.py            (Python)
11. no11_wm-gm-fc_3d.py             (Python)
12. no12_wm-gm-fc_heatmap.py        (Python)
```

Each script has a USER SETTINGS block -- verify SUBJECTS and BASE_DIR before running.

---

### Inputs and Prerequisites

Raw subject data required:
- `sub-XXXX_dti.nii.gz` (or .dcm) + .bvec + .bval
- `sub-XXXX_t1.nii`
- `sub-XXXX_rest.nii`

Templates and atlases:
- AALv3 atlas (`ROI_MNI_V7_1mm.nii`, `AAL3v1.nii.txt`)
- NTU-DSI-122 template
- DARTEL flow fields:
    `iy_NCKU_336Ss_6.nii`                              (MNI->SST)
    `u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`         (SST->subject)

GM pipeline outputs required:
- `gm/anat/segmentation/c1{subj}_t1.nii`
- `gm/derived/dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `gm/derived/fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `gm/func/derived/fc_results/{subj}_R.npz`
- `gm/anat/derived/dist_results/{subj}_W_a.npz`

---

### Step 1: no1_recon-all.sh

FreeSurfer cortical reconstruction and thalamic nuclei segmentation.

USER SETTINGS: BASE_DIR, RAW_DIR, DERIV_DIR, SUBJECT_LIST

Input: `rawdata/{subj}/ses001/anat/{subj}_t1.nii`

What it does:
1. Sets SUBJECTS_DIR to `{DERIV_DIR}/{subj}/wm/freesurfer`
2. Runs `recon-all -s {subj} -i {T1} -all`  (~3-3.5 hours per subject)
3. Runs `segmentThalamicNuclei.sh {subj}`
   (required because dmrirc has `usethalnuc = 1`)

Output: `wm/freesurfer/{subj}/`  (full FreeSurfer directory)

---

### Step 2: no2_{subj}_dmrirc  (Manual)

TRACULA configuration file -- must be written manually per subject before running no3.
Save as: `wm/tracula/{subj}_dmrirc`

Key fields to edit per subject:

Field                          | Description
-------------------------------|----------------------------------------------------
SUBJECTS_DIR                   | Path to wm/freesurfer/
dtroot                         | Path to wm/freesurfer/ (TRACULA outputs here)
dob0                           | B0 correction: 0=none, 1=field map, 2=reverse-PE
subjlist                       | ( {subj} {subj} ) -- repeated for fwd + rev PE
dcmroot                        | Root of DTI source data
dcmlist                        | ( dti/IM-0009-0001.dcm dti_r/IM-0008-0001.dcm )
echospacing, pedir, epifactor  | Reverse-PE parameters (when dob0=2)
doeddy                         | 1 to run eddy-current correction
usethalnuc                     | 1 (requires segmentThalamicNuclei.sh from Step 1)
pathlist                       | List of 40 tracts (lh.cst and rh.cst excluded)

The original example file is saved as `dmrirc.example`.

---

### Step 3: no3_trac-all.sh

Runs TRACULA in three phases for each subject.

USER SETTINGS: ANTSPATH, BASE_DIR, SUBJECTS

Prerequisites: ANTs must be installed; set ANTSPATH correctly.
Config file expected at: `{BASE_DIR}/{subj}/wm/tracula/{subj}_dmrirc`

Phases (run sequentially; subject is skipped if any phase fails):
1. `trac-all -prep -c {subj}_dmrirc`  -- preprocessing (registration, masking)
2. `trac-all -bedp -c {subj}_dmrirc`  -- BEDPOSTX (fiber orientation modelling)
3. `trac-all -path -c {subj}_dmrirc`  -- pathway reconstruction

Expected outputs in `wm/freesurfer/{subj}/`:
- `dmri/`          -- preprocessed DWI
- `dmri.bedpostX/` -- BEDPOSTX results
- `dpath/`         -- reconstructed tract directories (one per pathway)
- `dlabel/`        -- label volumes

---

### Step 4: no4_endpoints_coordinates.py

Extracts voxel (ijk) and RAS (xyz) center-of-gravity coordinates for each tract endpoint.

USER SETTINGS: SUBJECTS, BASE_DIR

Input:  `wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz`

Method: `fslstats -C` (intensity-weighted voxel COG)
        `fslstats -c` (RAS COG)

Output: `wm/derived/endpoints/endpt{1,2}/{subj}_{tract}_endpt{1,2}.csv`
        Columns: tract, i, j, k, x, y, z

Note: Returns a single coordinate per endpoint (center of gravity), not all active voxels.
      no7_aal_summary.py reads `endpt.pd.nii.gz` directly for full voxel-level AAL lookup.

---

### Step 5: no5_endpoints_size.py

Counts active voxels (pd > 0) in each tract endpoint probability density map.

USER SETTINGS: SUBJECTS, BASE_DIR, PD_THRESHOLD (default 0)

Input:  `wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz`

Output: `wm/derived/endpoints/{subj}_endpoints_size.csv`
        Columns: subject, tract, endpoint, n_voxels

---

### Step 6: no6_pathstats.py

Parses `pathstats.overall.txt` for each tract and aggregates scalar WM metrics per subject.

USER SETTINGS: SUBJECTS, BASE_DIR, VERBOSE

Input:  `wm/freesurfer/{subj}/dpath/{tract}/pathstats.overall.txt`

Output: `wm/derived/pathstats/{subj}_pathstats.csv`
        Columns: tract, NumStreamlines, FA_Avg, MD_Avg, AD_Avg, RD_Avg, ...

---

### Step 7: no7_aal_summary.py

For each tract endpoint, scans all active voxels in the DWI-space AAL atlas
and computes the top-3 AAL regions by voxel overlap percentage.

USER SETTINGS: SUBJECTS, BASE_DIR,
               LUT_PATH (path to AAL3v1.nii.txt),
               TOPN (default 3),
               PD_THRESHOLD (default 0)

Input:
- `wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz`
- `gm/derived/dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `AAL3v1.nii.txt`  (LUT)

Output: `wm/derived/aal_summary/{subj}_aal_summary.csv`
        Columns: tract, endpoint,
                 top1_aal_id, top1_aal_name, top1_percent,
                 top2_aal_id, top2_aal_name, top2_percent,
                 top3_aal_id, top3_aal_name, top3_percent

---

### Step 8: no8_aal_heatmap.py

Generates AAL-AAL weighted connectivity matrices and heatmap images per WM metric.

USER SETTINGS: SUBJECTS, BASE_DIR, METRICS (default: FA, AD, MD, RD)

Input:
- `wm/derived/pathstats/{subj}_pathstats.csv`
- `wm/derived/aal_summary/{subj}_aal_summary.csv`

Weight formula:
  `metric_avg x (top_i_percent/100) x (top_j_percent/100)`
  for all top-1~3 endpoint region pairs per tract.

Output: `wm/derived/aal_summary/heatmap/`
- `aal_aal_{metric}_matrix_{subj}.csv`       -- symmetrised AAL x AAL matrix
- `aal_edge_details_{metric}_{subj}.csv`     -- per-edge detail (roi_u, roi_v, weight, tract)
- `aal_aal_{metric}_heatmap_{subj}.png`

---

### Step 9: no9_aal_rank_edge.py

Ranks individual ROI-ROI edges by weighted connectivity score per metric (descending).

USER SETTINGS: SUBJECTS, BASE_DIR, METRICS

Input:  `wm/derived/aal_summary/heatmap/aal_edge_details_{metric}_{subj}.csv`

Output: `wm/derived/aal_summary/rank_edge/aal_edge_rank_{metric}_{subj}.csv`
        Columns: roi_u, roi_v, weight, tract  (sorted by weight descending)

---

### Step 10: no10_aal_rank_roi.py

Ranks AAL regions by total weighted connectivity score across all tract edges per metric.

USER SETTINGS: SUBJECTS, BASE_DIR, METRICS

Input:  `wm/derived/aal_summary/heatmap/aal_edge_details_{metric}_{subj}.csv`

Score:  Sum of all edge weights involving each ROI (as either endpoint).

Output: `wm/derived/aal_summary/rank_roi/aal_roi_rank_{metric}_{subj}.csv`
        Columns: roi, score

---

### Step 11: no11_wm-gm-fc_3d.py

Generates 3D scatter plots of WM metric x GM geodesic distance x FC.
One point = one GM voxel pair connected via a WM tract.

USER SETTINGS: SUBJECTS, BASE_DIR, LUT,
               METRICS,
               MAX_VOXELS  (voxels sampled per AAL region, default 200),
               ELEV, AZIM, DPI

Input:
- `wm/derived/endpoints/endpt{1,2}/{subj}_{tract}_endpt{1,2}.csv`  (endpoint COG ijk)
- `wm/derived/pathstats/{subj}_pathstats.csv`                       (FA, RD, MD, AD)
- `gm/derived/dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `gm/derived/fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii`
- `gm/func/derived/fc_results/{subj}_R.npz`
- `gm/anat/derived/dist_results/{subj}_W_a.npz`
- `gm/anat/segmentation/rc1*.nii`  (for GM voxel set construction)

What it does:
1. Looks up AAL region for each tract endpoint in DWI space
   (3x3x3 neighbourhood fallback if endpoint voxel is background)
2. Builds fMRI-space region groups from fMRI AAL + c1 voxel set
3. Samples up to MAX_VOXELS voxels per region;
   extracts FC and geodesic distance for all cross-region pairs
4. Assigns WM metric value to each pair from pathstats
5. Saves voxelpair CSV; generates static PNG and interactive HTML (if plotly installed)

Output: `wm-gm-fc/3d/`
- `{subj}_voxelpair_{metric}.csv`     -- columns: fc, dist, {metric}, tract, idx_u, idx_v
- `{subj}_tract_3d_{metric}.png`
- `{subj}_tract_3d_{metric}.html`     (if plotly installed)

Note: MAX_VOXELS caps sampling per AAL region to avoid memory issues.
      The voxelpair CSV is required by no12.

---

### Step 12: no12_wm-gm-fc_heatmap.py

Generates voxel x voxel heatmaps for FC, geodesic distance, and WM radial diffusivity (RD).

USER SETTINGS: SUBJECTS, BASE_DIR,
               SAMPLE_N  (voxels sampled per axis, default 500),
               DPI

Input:
- `gm/func/derived/fc_results/{subj}_R.npz`
- `gm/anat/derived/dist_results/{subj}_W_a.npz`
- `wm-gm-fc/3d/{subj}_voxelpair_RD.csv`  (from no11)

Output: `wm-gm-fc/heatmaps/`
- `heatmap_A_fc_{subj}.png`    -- FC matrix (RdBu_r colormap)
- `heatmap_B_dist_{subj}.png`  -- Geodesic distance (Greys)
- `heatmap_C_rd_{subj}.png`    -- RD voxel x voxel (Blues)

The same random seed (42) is used across all three panels for consistent voxel sampling.

---

## Summary

The WM pipeline transforms subject-specific diffusion data into anatomically grounded,
GM-referenced structural connectivity representations. By integrating TRACULA tracts
with GM functional connectivity and AAL parcellation in subject space, the pipeline
enables fine-grained, interpretable structure-function analyses at the individual level.

After all steps, verify that the directory tree matches the expected structure
(see Directory Structure section above).