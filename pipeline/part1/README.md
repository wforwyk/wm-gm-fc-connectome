# Part 1: subject-level processing

See the [pipeline README](../README.md) for the cross-stage execution order,
40-tract extraction / 39-tract group analysis, participant selection, and settings.
Keep the full `part1/` tree: Python, MATLAB, and shell scripts use its participant
selection helpers. Part 1 has no default participant exclusion; configure the
participant manifest for your run and configure Part 2 selection in `cohort_settings.py`.

## GM preprocessing: SPM MAT batches

| Step | File | Role |
| --- | --- | --- |
| no00 | `gm/pt1_gm_no00_setting_directory_data.sh` | Source → rawdata + numeric derivative layout |
| no01 | `gm/pt1_gm_no01_setting_preprocessing_jobfile.mat` | Six-stage SPM preprocessing job template |
| no02 | `gm/pt1_gm_no02_run_preprocessing_batchfile.mat` | SPM Run Jobs batch supplying subject inputs to no01 |
| helper | `gm/apply_bias_field_image.m` | Called by no01's bias-field application job; required even with MAT batches |
| no03 | `gm/pt1_gm_no03_denoise.m` | aCompCor + motion regressors, SPM first-level model and residuals |
| no04 | `gm/pt1_gm_no04_detrend_bandpass.m` | Linear detrending and 0.009–0.08 Hz band-pass |
| no05 | `gm/pt1_gm_no05_masking_binarise.m` | GM masking and preliminary binary mask |
| no06 | `gm/pt1_gm_no06_corr_mat_dist.py` | Final BnB mask, FC matrix, geodesic distances and voxel index |
| no07 | `gm/pt1_gm_no07_fc_distance_summary.py` | Subject FC/distance summaries |

No01 contains realignment, mean-functional segmentation, bias-field application,
slice timing, T1-to-mean-functional coregistration, and T1 segmentation. Its unresolved
inputs are supplied by no02. **Do not run the unconfigured no01 template directly.**

### Configure the MAT batches in MATLAB + SPM

1. Run GM no00 and verify the numeric subject paths under `BATCH_DIR`.
2. Add SPM and `pipeline/part1/gm` to the MATLAB path. Open no01 in the SPM Batch
   Editor. Set the tissue probability maps for **both** Segment jobs to your SPM
   installation's `tpm/TPM.nii,1` … `,6`. Re-select `apply_bias_field_image` in
   the Call MATLAB Function job so the saved function handle resolves on your
   machine. The distributed helper is required; do not remove it.
3. Verify acquisition settings in Slice Timing. The supplied
   defaults are **TR=2 s, TA=1.9375 s, 32 slices, order `[2:2:32 1:2:31]`,
   reference slice=2**. Change them for another acquisition. Preserve the
   `cfg_dep` connections between jobs.
4. Save a configured no01 MAT job. Open no02 in the Batch Editor and point
   `Run Jobs → Jobs` to that configured file. The distributed no02 contains
   **one example subject `0001`**, with 240 functional-volume references and
   one T1 reference; replace these references with your input files.
5. Replace/extend `Inputs` with each selected subject's functional scans
   `BATCH_DIR/{id}/gm/func/preprocessing/{id}_rest.nii,1…N_VOLS` and T1
   `BATCH_DIR/{id}/gm/anat/segmentation/{id}_t1.nii,1`. Set the generated-job
   save directory to an existing writable location, and review the Run Jobs
   missing-input policy (the supplied value is `skip`). Save the configured no02.
6. Run the configured no02 from the Batch Editor, or:

   ```matlab
   addpath('/path/to/your/spm12');
   addpath('/path/to/your/pipeline/part1/gm');
   spm('defaults', 'fmri');
   spm_jobman('initcfg');
   job = load('/path/to/your/configured_no02.mat', 'matlabbatch');
   spm_jobman('run', job.matlabbatch);
   ```

7. Check outputs before GM no03. Set no03 `TR`, `N_VOLS`, `N_SLICES`, and
   `ONSET_SLICE` to the same acquisition and reference slice; set no04 `TR`
   identically. Set `N_VOLS` to the number of functional volumes in your data.

Run and check one participant in SPM before processing the full cohort.

### GM output contract

All paths below are relative to `BATCH_DIR/{id}`.

| Producer | Output | Consumer |
| --- | --- | --- |
| MAT no01/no02 | `gm/func/preprocessing/mean{id}_rest.nii` | Template no03/no06 |
| MAT no01/no02 | `gm/func/preprocessing/c1mean{id}_rest.nii`, `c2mean{id}_rest.nii` | Part 2 no08/no07 |
| MAT no01/no02 | `gm/func/preprocessing/abr{id}_rest.nii`, `rp_{id}_rest.txt` | GM no03 |
| MAT no01/no02 | `gm/anat/segmentation/c1{id}_t1.nii` and c2/c3 tissue maps | GM no03/no04/no05 |
| GM no03 | `gm/func/model/1st_level/Res_*.nii` | GM no04; moves residuals to its `residuals/` subdirectory |
| GM no04 | `gm/func/derived/denoised_detrended_rest_{id}.nii` | GM no05/no06 |
| GM no05 | `gm/func/derived/mdenoised_detrended_rest_{id}.nii`, `bmdenoised_detrended_rest_{id}.nii` | GM no06 |
| GM no06 | `gm/func/derived/bnbmdenoised_detrended_rest_{id}.nii`, `{id}_vset.npz` | Endpoint reslicing, WM no11 and Part 2 |
| GM no06 | `gm/func/derived/fc_results/{id}_R.npz` | WM no11/no12, Part 2 |
| GM no06 | `gm/anat/derived/dist_results/{id}_W_a.npz`, `{id}_A.npz` | GM no07, WM no11/no12, Part 2 |

The final BnB mask is made in **no06**, not no05. R/W_a/vset/BnB must share
one C-order voxel index and be regenerated together.

## WM extraction and summaries

| Step | File | Dependency / output |
| --- | --- | --- |
| no01 | `wm/pt1_wm_no01_recon-all.sh` | Raw T1 → FreeSurfer subject |
| no02 | `wm/pt1_wm_no02_subjectid_dmrirc.example` | Copy and configure one dmrirc per subject; exactly 40 tracts, retaining MCP/rh.atr/acomm |
| no03 | `wm/pt1_wm_no03_trac-all.sh` | dmrirc → TRACULA dwi.nii, dpath, endpoint PD and pathstats |
| no04 | `wm/pt1_wm_no04_endpoints_coordinates.py` | Endpoint coordinate CSVs |
| no05 | `wm/pt1_wm_no05_endpoints_size.py` | Endpoint sizes |
| no06 | `wm/pt1_wm_no06_pathstats.py` | `{id}_pathstats.csv` |
| no07 | `wm/pt1_wm_no07_aal_summary.py` | DWI-space AAL + **DSI Studio `AAL_LUT`** → `{id}_aal_summary.csv` |
| no08 | `wm/pt1_wm_no08_aal_heatmap.py` | AAL weighted connectivity matrices/heatmaps |
| no09/no10 | `wm/pt1_wm_no09_aal_rank_edge.py`, `wm/pt1_wm_no10_aal_rank_roi.py` | Edge/ROI ranks |
| no11 | `wm/pt1_wm_no11_wm-gm-fc_3d.py` | R/W_a/vset + endpoint coordinates + DWI/fMRI AAL + **DSI Studio `AAL_LUT`** → `wm-gm-fc/3d/{id}_voxelpair_RD.csv` and visualizations |
| no12 | `wm/pt1_wm_no12_wm-gm-fc_heatmap.py` | no11 voxelpair_RD + R/W_a → heatmaps |

Set `SOURCE_SESSION` and `DARTEL_TEMPLATE_ID` consistently in WM no07/no11.
No12 inherits those atlas/LUT choices through the no11 output; it does not load a
LUT itself. Do not point no07/no11 to the canonical AAL3 LUT. Exclude MCP only in
Part 2; do not remove it from the dmrirc or subject-level extraction.

## Template and registration steps

| Step | File | Role |
| --- | --- | --- |
| no00 | `template/pt1_template_no00_copy_u_rc1.sh` | Copy externally generated cohort DARTEL flow fields |
| no01 | `template/pt1_template_no01_NTU-DSI-122_manual.md` | Paired DSI Studio atlas/LUT export and MNI → SST preparation |
| no02 | `template/pt1_template_no02_SST_to_sub.m` | SST AAL → subject space using DARTEL |
| no03 | `template/pt1_template_no03_coregister_to_dwi_fmri.m` | AAL reslice onto mean-functional and DWI grids |
| no04 | `template/pt1_template_no04_verification.sh` | AAL grid checks; optional endpoint checks after no05 |
| no05 | `template/pt1_template_no05_coregister_endpoints_to_fmri.m` | Continuous endpoint PD → final BnB grid, producing `fmri_endpt{1,2}.pd.nii` |
| no06 | `template/pt1_template_no06_verify_dwi_fmri_alignment.sh` | Independent intensity/header alignment QC and overlays |
| no07 | `template/pt1_template_no07_yeo7_mni_to_sst.m` | Categorical Yeo7 → SST for Part 2 no11 |

Template no03/no05 are reslice operations and depend on previously
verified alignment. Neither matching grids nor a correct LUT establishes anatomical
registration quality.
