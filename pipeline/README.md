# WM–GM–FC Connectome Pipeline

Deployment-ready source package for a two-part neuroimaging workflow that
relates white-matter tract properties to grey-matter voxel-wise functional
connectivity (FC) and geodesic distance in individual subject space.

This package contains scripts and documentation only. It does not include
participant data, templates, atlas files, TRACULA configuration files, output
matrices, figures, or any local working material.

## Before you run anything

Read every script's `USER SETTINGS` block. Paths, subject IDs, sessions, atlas
assets, and tool locations are deliberately placeholders such as
`/path/to/your/project`. Do not replace paths outside those settings blocks
unless you have also checked every downstream input/output contract.

Inspect your own raw-data layout first. This deployment uses the following
default convention:

```text
/path/to/your/project/
  rawdata/
    sub-0001/ses-01/anat/sub-0001_t1.nii
    sub-0001/ses-01/func/sub-0001_rest.nii
  derivatives/
    batch/0001/                    # numeric subject ID, no "sub-" prefix
    template/AAL3/
    template/sst_atlas/
    pt2_group_results/
```

Raw data retain the BIDS-style `sub-xxxx` identifier. The initial setup script
copies each subject into `derivatives/batch/xxxx`, where all later Part 1 and
Part 2 scripts use the numeric ID only. Change `SOURCE_SESSION`,
`RAW_SESSION`, source file patterns, and subject lists to match your data.

## Requirements

- MATLAB with SPM12 and its TPM file
- FreeSurfer 7.2 or later, including TRACULA
- FSL (`fslinfo`, `fslstats`)
- ANTs (needed by `trac-all`)
- DSI Studio for the manual template-preparation step
- Python 3 with `numpy`, `scipy`, `pandas`, `nibabel`, and `matplotlib`
  (`statsmodels` is optional for the Part 2 sensitivity model)

You must supply and configure the AAL3 atlas assets, DARTEL flow fields,
SST-space AAL image, TRACULA per-subject configuration, and participant data.

## Execution order

```text
Part 1, per subject
  GM:       pt1_gm_no00 → pt1_gm_no01_02 → no03 → no04 → no05 → no06 → no07
  Template: pt1_template_no00 → no01 (manual, once) → no02 → no03 → no04 → no05 → no06 (QC)
  WM:       pt1_wm_no01 → no02 (configure per subject) → no03 → no04 → no05 → no06 → no07 → no08 → no09 → no10 → no11 → no12

Part 2, group level
  pt2_no01
  Aim 1: pt2_no02 → pt2_no03
  Aim 2: pt2_no04 → pt2_no05 → pt2_no06
         pt2_no07 → pt2_no08 → pt2_no09 → pt2_no10
```

`part1/README.md` gives the subject-level input/output contracts. The manual
template step is in `part1/template/pt1_template_no01_NTU-DSI-122_manual.md`.
The WM–GM visualization steps (`pt1_wm_no11` 3D scatter and
`pt1_wm_no12` heatmaps) are included. Paper-figure renderers are not included.

## Settings that must agree across scripts

| Setting | Where it matters |
| --- | --- |
| `BATCH_DIR` / `BASE_DIR` | All Part 1 and Part 2 scripts; point to `derivatives/batch` |
| `SOURCE_SESSION` / `RAW_SESSION` | `pt1_gm_no00`, `pt1_wm_no01`, and template flow-field scripts |
| `DARTEL_TEMPLATE_ID` | Template scripts, WM AAL scripts, and Part 2 AAL readers |
| AAL3 LUT and border files | `pt2_no01` and scripts reading AAL regions |
| `TRACT_SUFFIX` | All endpoint and tract-matching steps; keep the TRACULA output suffix unless your data differ |
| Subject IDs | Numeric in `derivatives/batch`; BIDS-style only in `rawdata` |

The string `YOUR_DARTEL_TEMPLATE` is a required placeholder. Replace it with
the identifier embedded in your DARTEL flow-field filenames, and use the same
value in every script that exposes `DARTEL_TEMPLATE_ID`.

## Scientific and data-integrity constraints

- `R.npz`, `W_a.npz`, `vset.npz`, and the BnB mask must retain the identical
  C-order non-zero voxel ordering. Never regenerate only one of them.
- FC is stored as Pearson *r*. Apply Fisher-Z before averaging whenever an
  analysis requires a transformed mean.
- Geodesic distance is in millimetres and is restricted to the final GM voxel
  set.
- AAL labels are discrete: atlas resampling must use nearest-neighbour
  interpolation. Endpoint path-density maps are continuous and use trilinear
  interpolation.
- The subject, not the voxel, is the unit of group-level inference.

## Validation before scientific use

Run this package first on a controlled test subject and verify every expected
input/output path. Before using Aim 2, run
`part1/template/pt1_template_no06_verify_dwi_fmri_alignment.sh` across the
analysis subjects; inspect its header-versus-6-DOF-FLIRT residual and both
overlays before accepting header-only endpoint reslicing. Use the primary
5%-of-positive-maximum endpoint-PD threshold consistently in `pt2_no04` and
`pt2_no08`; preserve outputs deliberately before a threshold sensitivity run.
Part 2 is active research; its outputs require study-specific QC and
statistical review.

## Package contents

```text
part1/gm/        GM preprocessing, FC, and geodesic-distance scripts
part1/template/  atlas and endpoint reslicing scripts plus manual guide
part1/wm/        FreeSurfer/TRACULA and WM–GM integration scripts
part2/            group-level analysis scripts
```
