# WM–GM–FC connectome pipeline

Subject-space white-matter tract processing, grey-matter voxel-wise functional
connectivity (FC), and participant-level group inference. Part 1 creates subject
outputs; Part 2 uses those outputs and atlas reference tables. Supply your own input data and atlas/template files.

## Requirements

| Component | Requirement / use |
| --- | --- |
| MATLAB + SPM12 | MAT batches, preprocessing, DARTEL, reslicing |
| FreeSurfer | TRACULA 42-tract atlas (requires ≥7.2), recon-all |
| FSL | fslstats and template/alignment QC |
| ANTs | TRACULA processing |
| DSI Studio | AAL label image **and its paired LUT** export |
| Python | 3.10+; numpy, scipy, pandas, nibabel, matplotlib |
| Optional Python packages | plotly for interactive WM no11; statsmodels for no10 sensitivity model |

## Configuration

Edit each script's `USER SETTINGS`. `/path/to/your/...` values must be
replaced before execution. `BATCH_DIR` is the directory containing numeric subject
folders, not the project root. The default session/template tokens are examples;
use the exact tokens in your own DARTEL filenames.

Python/MATLAB/shell Part 1 scripts accept `SUBJECTS`, `SUBJECTS_TXT`, and
`EXCLUDE_SUBJECTS`. A nonempty text-file path overrides the inline list; one ID per
line, optional `sub-` prefix, blank/comment lines ignored. An empty list discovers
numeric directories. Keep a complete `part1/` directory so its selection helpers
remain available. MAT batch subject inputs are configured in the SPM Batch Editor,
as described in Part 1; they do not import Python settings.

Part 2 participant and group stages share [cohort_settings.py](part2/cohort_settings.py).
Set `SUBJECTS_TXT` there to your cohort manifest for reproducible selection and
set `EXCLUDE_SUBJECTS` to your own exclusions (empty by default). These settings
apply across participant processing and group-table readers. Empty selection uses all available inputs. Atlas-only no01/no11 are not participant-selection stages.

### Settings that must agree

| Setting | Stages / contract |
| --- | --- |
| `BATCH_DIR` | All subject stages, MAT input paths, dmrirc subject paths |
| `SOURCE_SESSION`, `DARTEL_TEMPLATE_ID` | Template no00/no02/no03/no04; WM no07/no11; Part 2 no02/no04/no05/no08/no09 |
| `AAL_LUT` | WM no07/no11 and Part 2 no01: same DSI Studio export LUT |
| `ORIGINAL_AAL_TXT`, `BORDER_MAT` | Part 2 no01: canonical name/category crosswalk and canonical border geometry only |
| `TRACT_SUFFIX` | Template no04/no05 and Part 2 no02/no04/no08; default `_avg16_syn_bbr` |
| `QC_EXCLUDE` | Part 2 no02/no03/no04/no08: bilateral CST and MCP only; retain rh.atr/acomm |
| `ENDPOINT_RELATIVE_THRESHOLD` | Part 2 no04/no08: 0.05 of positive endpoint-map maximum; 0.01/0.10 require separate sensitivity output locations |
| `WM_PROB_THRESHOLDS`, `PRIMARY_WM_PROB_THRESHOLD` | no07 produces 0.5/0.7/0.9 maps; no08 primary setting 0.5 generates the matching filename |
| `MIN_MATCHED_VOXELS` | no09/no10: 10; no10 also applies matching-balance requirements |
| Acquisition parameters | no01 MAT slice-timing settings ↔ GM no03 `TR`, `N_VOLS`, `N_SLICES`, `ONSET_SLICE`; GM no04 `TR` |
| Participant selection | Part 1 local selection settings; Part 2 shared cohort manifest and exclusions |

## Execution order and cross-stage dependencies

```text
GM:  no00 → no01 MAT (job configuration) → no02 MAT (batch execution)
     → no03 → no04 → no05 (GM mask) → no06 (BnB, R, W_a, vset) → no07
WM:  no01 recon-all → no02 dmrirc → no03 trac-all (dwi.nii and dpath)
TPL: no00 (copy external DARTEL flow fields) → no01 manual atlas/LUT export
     → no02 → no03 [requires GM mean image + WM no03 dwi.nii]
     → no04 QC
TPL: no05 endpoint reslice [requires GM no06 BnB + WM no03 dpath + alignment QC]
     → no04 endpoint QC; no06 independent anatomical-alignment QC
WM:  no04 → no05 → no06 → no07 [requires TPL no03 DWI-space AAL + paired LUT]
     → no08 → no09/no10; no11 [requires GM no06] → no12
Part 2: no01 reference tables [export LUT + canonical crosswalk/border file]
     Region-pair FC: no02 → no03
     Endpoint FC: no04 → no05 → no06
     Matched: no07 [GM c2mean + BnB] → no08 → no09 → no10
     Networks: TPL no07 → no11 [SST AAL + SST Yeo] → no12 → no13
               no12 also requires no09 + no10 eligibility/subject tables
```

GM no05 writes `bmdenoised_…`; **GM no06 writes the final `bnbmdenoised_…` mask**
used by endpoint reslicing and Part 2. Matching dimensions alone do not demonstrate
DWI-to-fMRI anatomical alignment; review the overlays and alignment QC before
using endpoint reslices.

DARTEL flow fields are generated outside this package during cohort T1 template
construction. Template no00 only copies existing fields named
`u_rc1sub-{id}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii`.

## AAL identity and reference tables

Keep the DSI Studio label image and LUT together throughout the MNI → SST → native
space transformations. WM no07 uses the exported IDs for endpoint assignments; WM no11 uses
them for voxel-pair summaries. Part 2 no01 joins **ROI names** to the original AAL3
LUT to obtain canonical categories, then writes categories using **export IDs**.
The canonical border matrix's six-neighbour adjacency is translated to export IDs;
it is not recomputed in the warped SST/subject geometry. Subsequent stages read
`aal3_region_category.csv` and `aal3_adjacent_pairs.csv` in that namespace.

Use the paired export LUT (`DSIstudio/all_ROIs_AAL.txt`) in WM no07/no11 and
Part 2 no01. All endpoint summaries and reference tables must use the same
export-ID namespace.

## Part 2 outputs and statistical tests

All default group output directories are under `derivatives/results/part2/`.

| Stage | Main input → output |
| --- | --- |
| no01 | Export LUT + canonical LUT/border → `aal3_region_category.csv`, `aal3_adjacent_pairs.csv` in template/AAL3 |
| no02 | R/W_a/BnB + fMRI AAL + WM aal_summary/pathstats → `{id}_wm_present_absent_pairs.csv` in `no02_wm_fc` and tract eligibility audits |
| no03 | no02 pairs → pair/tract tables, subject effects, distance-adjusted statistics and figures in `no03_summary_stats` |
| no04 | fMRI endpoint PD + AAL/BnB/aal_summary → `{id}_endpoint_voxel_labels.npz` in `no04_endpoint_voxels` |
| no05 | no04 labels + R/W_a + adjacency → `{id}_aim2_endpoint_fc.csv` in `no05_endpoint_fc` |
| no06 | no05 rows → participant-level endpoint/target contrasts in `no06_endpoint_stats` |
| no07 | c2mean + BnB → WM probability-threshold masks and `wm_boundary_distance_c2thr0p5_{id}.nii.gz` under each subject's `gm/anat/derived/aim2_wm_boundary_distance` |
| no08 | no07 distance + c1mean + endpoints/AAL/BnB → `{id}_aim2_endpoint_covariates.npz` in `no08_endpoint_covariates` |
| no09 | no08 covariates + R/W_a/adjacency → matched FC and matching-balance CSVs in `no09_matched_endpoint_fc` |
| no10 | no09 matched rows/balance → eligibility, subject interaction, balance and statistics tables in `no10_matched_endpoint_stats` |
| no11 | SST AAL/Yeo + category table → `aal3_yeo7_overlap_lookup.csv` and QC in `no11_yeo7_aal_lookup` |
| no12 | no09 + no10 eligibility + no11 → participant/tract/system/network effects in `no12_tract_network_prepare`, reconciled to no10 |
| no13 | no12 effects → participant-level tests in `no13_tract_network_stats` |

No03 uses a within-participant WM-present minus WM-absent contrast after
participant-specific FC~log(geodesic-distance) residualization, followed by a
one-sample t-test across participants. Multiple
tracts supporting the same region pair are collapsed for the primary pair-level
analysis. The optional RD–FC screen is disabled by default
(`RUN_EXPLORATORY_RD_FC = False`), exploratory, and not a myelin-specific test.

The extraction set contains 40 TRACULA tracts: the 42-tract atlas excluding
`lh.cst` and `rh.cst`. The group-analysis set contains 39 tracts, excluding `mcp`.
MCP remains in Part 1 processing; `rh.atr` and `acomm` are included throughout.
Data eligibility can reduce the observed set in a particular contrast.

Pathway taxonomy distinguishes commissural, association, projection, and other.
Bilateral fornix is classified as other. MCP has the same classification but is
excluded from group analysis. All four classes enter the tract-system analyses.
TRACULA reference: Maffei et al. (2021), NeuroImage 245:118706.

In no03, each system contrast uses the participant's primary FC ~ log(distance)
fit and counts each region pair once within that system. All systems share the
participant's WM-absent comparison pool. Four distance-adjusted tests receive
BH-FDR correction; raw differences are descriptive only.

In no12, system effects include all no10-eligible endpoint units without a
cortical-pair restriction. Endpoint counts (`n_units`) recombine split rows within
each participant/tract; tracts then receive equal weight within each system.
No13 requires these no12 outputs, tests each of the four system effects against
zero with BH-FDR correction, and computes the commissural-minus-association contrast.

Yeo network analyses require cortical pairs with network labels across all four
classes. The pooled within-minus-between test, commissural-versus-association
network interaction, and four class-specific network contrasts form a secondary
BH-FDR family per mapping. The primary mapping uses maximum overlap; ≥50%
dominance is a sensitivity subset.

## Data consistency and QC

- R contains Pearson r. Fisher-Z is applied before averaging for inference.
- R, W_a, vset and BnB share one C-order voxel index; regenerate them together.
- Geodesic distances use voxel sizes in mm on the final GM voxel graph.
- Interpret atlas IDs using the paired export LUT and review atlas-label QC.
- The participant is the inferential unit; pooled voxel/tract rows are not independent participants.
- Check preprocessing logs, registration overlays and participant-level QC before
  running group analyses.
