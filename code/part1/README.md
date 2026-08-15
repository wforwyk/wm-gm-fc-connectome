# Part 1: subject-level pipeline

Part 1 creates the per-subject grey-matter, white-matter, atlas, FC, and
geodesic-distance products consumed by Part 2. Configure the `USER SETTINGS`
block in every script before running it on your compute environment.

## Subject and directory convention

The source convention is intentionally explicit:

```text
rawdata/sub-0001/ses-01/              # BIDS-style raw subject directory
derivatives/batch/0001/               # numeric-only processing directory
```

`pt1_gm_no00_setting_directory_data.sh` performs this conversion. It keeps
the raw `sub-0001` name, then copies files into the numeric `0001` batch
directory expected by the later scripts. Inspect your own subject directory,
session label, and T1/rest filenames first; edit that script's source paths
and `SOURCE_SESSION` / `RAW_SESSION` settings as needed.

## GM steps

| Order | Script | Primary output |
| --- | --- | --- |
| 0 | `gm/pt1_gm_no00_setting_directory_data.sh` | raw-data copy and batch directory tree |
| 1–2 | `gm/pt1_gm_no01_02_preprocessing.m` | SPM realignment, segmentation, bias correction, slice timing, and coregistration |
| 3 | `gm/pt1_gm_no03_denoise.m` | aCompCor/motion nuisance model and residuals |
| 4 | `gm/pt1_gm_no04_detrend_bandpass.m` | detrended, band-pass-filtered series |
| 5 | `gm/pt1_gm_no05_masking_binarise.m` | BnB analysis mask |
| 6 | `gm/pt1_gm_no06_corr_mat_dist.py` | `R.npz`, `W_a.npz`, and `vset.npz` |
| 7 | `gm/pt1_gm_no07_fc_distance_summary.py` | FC–distance QC summaries |

## Template steps

| Order | Script | Primary output |
| --- | --- | --- |
| 0 | `template/pt1_template_no00_copy_u_rc1.sh` | subject DARTEL flow field in `gm/derived` |
| 1 | `template/pt1_template_no01_NTU-DSI-122_manual.md` | SST-space AAL image (manual, once) |
| 2 | `template/pt1_template_no02_SST_to_sub.m` | native-T1 AAL image |
| 3 | `template/pt1_template_no03_coregister_to_dwi_fmri.m` | DWI- and fMRI-space AAL images |
| 4 | `template/pt1_template_no04_veritifaction.sh` | grid and label checks |
| 5 | `template/pt1_template_no05_coregister_endpoints_to_fmri.m` | fMRI-grid endpoint PD maps |

Set `DARTEL_TEMPLATE_ID` consistently wherever it appears. It is the
identifier in your own flow-field filename, not a directory path. The default
`YOUR_DARTEL_TEMPLATE` is deliberately non-runnable until configured.

## WM steps

| Order | Script | Primary output |
| --- | --- | --- |
| 1 | `wm/pt1_wm_no01_recon-all.sh` | FreeSurfer reconstruction |
| 2 | `wm/pt1_wm_no02_subjectid_dmrirc.example` | per-subject TRACULA configuration template |
| 3 | `wm/pt1_wm_no03_trac-all.sh` | TRACULA tractography |
| 4–6 | `wm/pt1_wm_no04`–`no06` | endpoint coordinates/size and tract statistics |
| 7–10 | `wm/pt1_wm_no07`–`no10` | AAL assignments, weights, and rankings |
| 11–12 | `wm/pt1_wm_no11`–`no12` | WM–GM–FC scatterplots and heatmaps |

Copy and configure `pt1_wm_no02_subjectid_dmrirc.example` for each subject as
`<subject>_dmrirc`. Its diffusion inputs and phase-encoding settings are
study-specific and must be verified against the acquisition protocol.

## Required checks

- Confirm that the DWI, fMRI, and T1 images are anatomically aligned before
  running endpoint reslicing.
- Keep the BnB mask, FC matrix, distance matrix, and voxel set together.
- Confirm every AAL output is on the intended reference grid; use nearest
  neighbour for label images.
- Review failed-subject reports and QC CSVs before sending products to Part 2.
