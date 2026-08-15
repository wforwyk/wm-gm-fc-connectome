# WM–GM–FC Connectome Pipeline (BrainWorld)

Research code for a two-part pipeline that combines **white-matter (WM) tract-based
structural information** with **grey-matter (GM) voxel-wise functional connectivity (FC)
and geodesic distance**, entirely in **individual subject space**.

Part 1 processes each subject and produces that subject's indices. Part 2 pools those
indices across subjects and runs the group-level analyses. Part 2 is **active research**
and is still being extended; Part 1 is stable.

---

## 1. Repository layout

| Path | Role | Tracked |
| --- | --- | --- |
| `part1/gm/` | GM preprocessing → FC matrix + geodesic distance matrix | yes |
| `part1/template/` | AAL atlas warping MNI → SST → subject T1 → fMRI/DWI, plus endpoint-PD reslicing into fMRI | yes |
| `part1/wm/` | FreeSurfer + TRACULA → endpoints → AAL integration → WM-GM-FC outputs | yes |
| `part2/` | Group-level integration and statistics (`pt2_no01`–`pt2_no10`) | yes |
| `archives/` | Superseded script versions (v0–v2, per-batch variants) | **no** (local only) |
| `../tutorial/` | Earlier teaching version of the pipeline with a sample subject | reference only |
| `../ppt/` | Slides and pipeline diagrams | — |

`archives/` and `../tutorial/` are **not** the pipeline. The canonical implementation is
`part1/` + `part2/`. `tutorial/` predates the current numbering (`no1_…` rather than
`pt1_wm_no01_…`) and is kept unchanged as an onboarding reference.

Every script carries a **`USER SETTINGS` block at the top**. Check `BASE_DIR` / `BATCH_DIR`
and the subject list before each run; nothing else normally needs editing.

---

## 2. Software requirements

| Software | Notes |
| --- | --- |
| MATLAB + SPM12 | SPM on the path (`/opt/spm`), TPM at `/opt/spm/tpm/TPM.nii` |
| FreeSurfer ≥ 7.2 | < 7.2 supports only 18 TRACULA tracts; the 42-tract atlas needs ≥ 7.2 |
| FSL | `fslstats`, `fslinfo` (WM `no04`, template verification) |
| ANTs | required by `trac-all`; set `ANTSPATH` |
| DSI Studio | manual template step only (`pt1_template_no01`) |
| Python 3 | numpy, scipy, pandas, nibabel, matplotlib; plotly optional; statsmodels optional (Part 2 `no10` sensitivity model) |

---

## 3. Execution order

```
Part 1
  GM        pt1_gm_no00 … no07          (per subject; must finish first)
  Template  pt1_template_no00 … no05    (no01 is manual, run once; rest per subject)
  WM        pt1_wm_no01 … no12          (needs GM + template outputs)
Part 2
  Reference pt2_no01                    (once, atlas level)
  Aim 1     pt2_no02 → pt2_no03
  Aim 2     pt2_no04 → pt2_no05 → pt2_no06          (primary)
            pt2_no07 → pt2_no08 → pt2_no09 → pt2_no10  (confounder-controlled)
```

The GM block must complete before the WM block: `c1`, the BnB mask, `R.npz`, `W_a.npz`
and `vset.npz` are all consumed by WM and Part 2 scripts.

---

## 4. Part 1 — subject-level processing

### 4.1 GM pipeline (`part1/gm`)

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `pt1_gm_no00_setting_directory_data.sh` | Copy T1 + resting-state into `rawdata/`, build the `derivatives/` tree, copy working images into place |
| 1–2 | `pt1_gm_no01_02_preprocessing.m` | SPM: realign (est+reslice) → segment mean func → apply bias field → slice timing → coregister T1 to func → segment coregistered T1. Produces `abr{subj}_rest.nii`, `rp_{subj}_rest.txt`, `mean{subj}_rest.nii`, `c1…c5{subj}_t1.nii`, `c1…c5mean{subj}_rest.nii` |
| 3 | `pt1_gm_no03_denoise.m` | Builds the nuisance model — **aCompCor** (5 PCs each from c2/c3 ROIs, thresholded and eroded at native resolution then resampled) plus **motion order 12** (6 parameters + temporal derivatives) — then SPM first-level specification and estimation with residuals |
| 4 | `pt1_gm_no04_detrend_bandpass.m` | Linear detrend of the residuals, then FFT band-pass **0.009–0.08 Hz** (REST-toolbox form) inside a permissive `c1 > 0.2` working mask |
| 5 | `pt1_gm_no05_masking_binarise.m` | `spm_mask` at **`c1 > 0.9`**, then binarise on "finite at every time point and not identically zero"; writes a connected-component QC row |
| 6 | `pt1_gm_no06_corr_mat_dist.py` | Final GM voxel set (mask ∧ finite ∧ var > 0, small components dropped), voxel × voxel Pearson FC, 26-connected graph with step lengths from the NIfTI header, blocked Dijkstra geodesic distance |
| 7 | `pt1_gm_no07_fc_distance_summary.py` | Streams the upper triangle of `R`/`W_a` in row blocks and accumulates exact FC–distance statistics; writes a per-subject summary row and a binned decay curve |

**Key GM outputs (per subject)**

```
gm/func/denoise/multiple_regressors.txt, nuisance_qc.txt
gm/func/model/1st_level/SPM.mat, beta_*.nii, residuals/Res_*.nii
gm/func/derived/denoised_detrended_rest_{subj}.nii
gm/func/derived/mdenoised_detrended_rest_{subj}.nii
gm/func/derived/bmdenoised_detrended_rest_{subj}.nii
gm/func/derived/bnbmdenoised_detrended_rest_{subj}.nii   <- the analysis GM mask
gm/func/derived/fc_results/{subj}_R.npz                  <- FC matrix   (n_gm x n_gm)
gm/func/derived/{subj}_vset.npz                          <- voxel coords in R/W row order
gm/anat/derived/dist_results/{subj}_A.npz                <- sparse adjacency
gm/anat/derived/dist_results/{subj}_W_a.npz              <- geodesic distance (mm)
```

Batch-level QC, written at `BASE_DIR`: `gm_no5_mask_qc.csv`, `gm_no6_qc.csv`,
`gm_no7_fc_distance_summary.csv`, `fc_distance_curves/{subj}_curve.csv`.

**Row-order contract.** `R.npz`, `W_a.npz`, `vset.npz` and the BnB mask all use the same
voxel ordering: C-order over the non-zero voxels of `bnbmdenoised_…nii`. Every downstream
script that indexes `R`/`W_a` depends on this. Do not regenerate one without the others.

**Current batch state.** `gm_no6_qc.csv` holds 334 subjects, `n_gm` ≈ 9,600–13,500 at
3.44 × 3.44 × 4.00 mm, all with a single connected graph component and
`frac_unreachable_pairs = 0`.

### 4.2 Template preparation (`part1/template`)

Propagates the AAL3 atlas into each subject's native spaces. `no01` is manual and runs
**once**; the rest run per subject.

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `pt1_template_no00_copy_u_rc1.sh` | Copy the DARTEL flow field `u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii` into `gm/derived/` |
| 1 | `pt1_template_no01_NTU-DSI-122_manual.md` | Manual: export AAL ROIs from DSI Studio, warp MNI → SST with `iy_NCKU_336Ss_6.nii` → `sst_all_ROIs_AAL.nii` (shared by all subjects) |
| 2 | `pt1_template_no02_SST_to_sub.m` | DARTEL `crt_iwarped` (K = 6, trilinear) SST → subject T1 → `wsst_all_ROIs_AAL_…nii` |
| 3 | `pt1_template_no03_coregister_to_dwi_fmri.m` | Coregister **reslice** (nearest neighbour, to preserve integer labels) into fMRI space (`fmri_` prefix, ref `mean{subj}_rest.nii`) and DWI space (`dwi_` prefix, ref `dmri/dwi.nii`) |
| 4 | `pt1_template_no04_veritifaction.sh` | Check AAL files and their reference grids; after step 5, set `VERIFY_ENDPOINTS=true` to also check every endpoint-PD output |
| 5 | `pt1_template_no05_coregister_endpoints_to_fmri.m` | Reslice each continuous TRACULA `endpt{1,2}.pd.nii.gz` map from DWI onto the BnB fMRI analysis grid (`fmri_` prefix, trilinear interpolation); required before Part 2 Aim 2 |

Nearest-neighbour interpolation in step 3 is deliberate: the atlas carries discrete AAL
label integers. Step 2 uses trilinear because a continuous DARTEL warp field is involved.

**Endpoint-PD reslicing (step 5).** TRACULA endpoint maps are continuous path-density
images in DWI space. Step 5 reslices them onto the BnB fMRI analysis grid with trilinear
interpolation and writes `fmri_endpt1.pd.nii` / `fmri_endpt2.pd.nii` in each tract's
dpath directory. It is reslice-only, not a newly estimated DWI→fMRI registration. It must
be run only after visual QC has confirmed that DWI and fMRI references are aligned in
subject space; matching image grids alone does not demonstrate anatomical alignment. This
grid match is necessary because Part 2 Aim 2 indexes FC and geodesic matrices using BnB
fMRI voxels.

### 4.3 WM pipeline (`part1/wm`)

40 tracts (the TRACULA 42 minus `lh.cst` / `rh.cst`, which were dropped at acquisition).

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `pt1_wm_no01_recon-all.sh` | `recon-all -all` (~3–3.5 h/subject) then `segmentThalamicNuclei.sh` (required by `usethalnuc = 1`) |
| 2 | `pt1_wm_no02_subjectid_dmrirc` | **Manual** TRACULA config per subject; reverse-PE B0 correction (`dob0 = 2`), eddy correction, 40-tract `pathlist` |
| 3 | `pt1_wm_no03_trac-all.sh` | `trac-all -prep` → `-bedp` → `-path`, aborting the subject on any failure |
| 4 | `pt1_wm_no04_endpoints_coordinates.py` | `fslstats -C` / `-c`: one intensity-weighted centre-of-gravity per endpoint, in voxel and RAS coordinates |
| 5 | `pt1_wm_no05_endpoints_size.py` | Active-voxel count per endpoint (`pd > 0`) — endpoint size proxy for QC |
| 6 | `pt1_wm_no06_pathstats.py` | Parse `pathstats.overall.txt` → `NumStreamlines`, `FA_Avg`, `MD_Avg`, `AD_Avg`, `RD_Avg`, … |
| 7 | `pt1_wm_no07_aal_summary.py` | Scan **all** active endpoint voxels against the DWI-space AAL atlas; report the top-3 regions with overlap percentages |
| 8 | `pt1_wm_no08_aal_heatmap.py` | AAL × AAL weighted matrices per metric, weight = `metric_avg × (pct_i/100) × (pct_j/100)` over the 3 × 3 top-region combinations |
| 9 | `pt1_wm_no09_aal_rank_edge.py` | Rank ROI–ROI edges by weight, descending |
| 10 | `pt1_wm_no10_aal_rank_roi.py` | Rank ROIs by summed edge weight (as either endpoint) |
| 11 | `pt1_wm_no11_wm-gm-fc_3d.py` | For each tract, sample up to `MAX_VOXELS` GM voxels in each endpoint region and emit every cross-region voxel pair as (FC, geodesic distance, WM metric); 3D scatter PNG + interactive HTML |
| 12 | `pt1_wm_no12_wm-gm-fc_heatmap.py` | Voxel × voxel heatmaps for FC, geodesic distance, and RD on a shared random voxel sample (seed 42) |

**Key WM outputs (per subject)**

```
wm/derived/endpoints/endpt{1,2}/{subj}_{tract}_endpt{1,2}.csv
wm/derived/endpoints/{subj}_endpoints_size.csv
wm/derived/pathstats/{subj}_pathstats.csv
wm/derived/aal_summary/{subj}_aal_summary.csv
wm/derived/aal_summary/heatmap/  rank_edge/  rank_roi/
wm-gm-fc/3d/{subj}_voxelpair_{metric}.csv, _tract_3d_{metric}.png/.html
wm-gm-fc/heatmap/heatmap_{A_fc,B_dist,C_rd}_{subj}.png
```

`pt1_wm_no11` requires `{subj}_vset.npz`, because the row index into `R`/`W_a` is defined
by that file and by nothing else.

---

## 5. Part 2 — group-level analyses (active research)

Part 2 reads the Part 1 per-subject products and writes into
`derivatives/pt2_group_results/no{k}_…/`. Subject is the unit of inference throughout:
every contrast is formed **within** subject, then tested against zero with a one-sample
t-test across subjects.

### 5.1 Atlas reference tables

`pt2_no01_region_category.py` runs once at the atlas level and produces the two tables
every later script reads:

- `aal3_region_category.csv` — category and `include` flag per AAL3v1 region.
  Included: `cortical`, `subcortical_limbic`, `thalamus_generic`, `thalamic_nuclei`,
  `acc_subdivision`. Excluded: `subcortical_basal`, `cerebellum`, `brainstem`, `n_acc`.
- `aal3_adjacent_pairs.csv` — physically adjacent region pairs, from 6-connectivity of
  the AAL border voxels in MNI space.

Tracts `rh.atr` and `acomm` are QC-excluded in every Part 2 script.

### 5.2 Aim 1 — does the presence of a WM tract predict higher GM FC?

| Step | Script | What it does |
| --- | --- | --- |
| 2 | `pt2_no02_wm_present_absent_fc.py` | Per subject, label every included AAL region pair **WM-present** (a QC-passing tract has its two endpoints' top-1 regions in that pair) or **WM-absent**; compute mean FC and mean geodesic distance over all cross-region GM voxel pairs; attach the tract's `RD_Avg` for WM-present pairs |
| 3 | `pt2_no03_stats.py` | Pool subjects; per-subject FC difference (present − absent) tested against zero; the same contrast per pathway type (commissural / projection / association, Maffei et al. 2021); per-subject Spearman FC ~ RD within WM-present pairs; violin, FC ~ distance, FC ~ RD and per-system figures |

### 5.3 Aim 2 — is FC elevated *specifically at* tract endpoint voxels?

The question is whether endpoint voxels prefer their WM-connected distant partner over
their physically adjacent neighbours, relative to non-endpoint voxels in the same AAL
region. Two chains answer it at increasing rigour.

**Primary chain (region-matched controls)**

| Step | Script | What it does |
| --- | --- | --- |
| 4 | `pt2_no04_endpoint_voxel_label.py` | Endpoint GM voxels = endpoint image ∩ BnB mask ∩ the endpoint's top-1 AAL region. Voxels of that region belonging to *no* tract endpoint become control candidates |
| 5 | `pt2_no05_endpoint_fc_comparison.py` | For endpoint voxels and for `N_CONTROL_REPEATS` random equal-sized control draws, compute mean Fisher-Z FC to the WM-connected distant region and to each adjacent region, plus the FC ~ log(distance) slope |
| 6 | `pt2_no06_endpoint_stats.py` | Aim 2.1 endpoint × target-type interaction; 2.2 endpoint vs control per target type; 2.3 endpoint vs control distance slope |

**Confounder-controlled chain (covariate-matched controls)**

| Step | Script | What it does |
| --- | --- | --- |
| 7 | `pt2_no07_wm_boundary_distance.py` | Euclidean distance (mm) from every voxel to the nearest WM voxel, WM defined by thresholding SPM `c2mean{subj}_rest.nii` at 0.5 (0.7 / 0.9 as sensitivity) |
| 8 | `pt2_no08_endpoint_voxel_covariates.py` | Same endpoint/control definition as `no04`, plus per-voxel WM-boundary distance and GM probability (`c1mean{subj}_rest.nii`) |
| 9 | `pt2_no09_matched_endpoint_fc.py` | 1:1 greedy nearest-neighbour matching on standardised (WM distance, GM probability) with a 2.0 caliper; same FC contrasts on the matched sets; writes an SMD balance table |
| 10 | `pt2_no10_matched_endpoint_stats.py` | Same three tests on balanced tract-ends only (`|SMD| ≤ 0.10`, ≥ 10 matched voxels), plus a covariate-adjusted subject-fixed-effect OLS with subject-clustered SEs as a sensitivity model |

---

## 6. Conventions

- **Subject IDs are numeric** under `derivatives/` (`1001`), while `rawdata/` uses the
  BIDS-style `sub-1001`. The template flow-field filenames keep the `sub-` form.
- **FC is stored as Pearson r.** Fisher-Z (`arctanh`, clipped to ±0.9999) is applied at
  analysis time, not at storage time.
- **Geodesic distance is in mm**, derived from the NIfTI voxel sizes, over a 26-connected
  graph restricted to the final GM voxel set.
- **Tract names** carry the TRACULA suffix `_avg16_syn_bbr` in Part 1 outputs; Part 2
  strips it (`TRACT_SUFFIX`) before matching.

---

## 7. Version-control policy

Track source code, configuration, and documentation only. Research data, subject-derived
files, generated matrices, results, figures, logs, local environments and credentials are
all ignored (see `.gitignore`); so is `archives/`, which is deliberately kept local.

Before sharing a script externally, replace machine- and institution-specific paths with
documented configuration variables. Several scripts still contain absolute
`/bml/projects/...` defaults.

---

## 8. Known issues

Open items in the current code, in rough priority order.

1. **Endpoint sets are inflated by the DWI → fMRI reslice.** `ENDPOINT_THRESHOLD = 0.0`
   in `pt2_no04` and `pt2_no08` keeps every voxel with any path density at all. Endpoint
   PD is an integer MCMC-sample count: measured on sub-1001, `lh.af` endpt1 has 472
   voxels above zero but only 136 above 5 % of its maximum, with a median count of 2 out
   of 62. Resampling 2.50 mm DWI voxels onto the 3.44 × 3.44 × 4.00 mm functional grid
   (3.03× larger) with trilinear interpolation then spreads the map further, and `> 0`
   keeps all of the bleed: across 84 tract-ends the endpoint volume inflates by a median
   of **2.75×** (IQR 2.39–2.95), falling to 1.95× at a 5 %-of-maximum threshold. This is
   a resolution-and-threshold effect, not a registration error — it is present even
   under perfect alignment. It both dilutes the endpoint set and shrinks the
   control-candidate pool in `pt2_no04`, biasing Aim 2 in the same direction twice.
   `ENDPOINT_THRESHOLD` should become a documented positive-valued setting, identical in
   `no04` and `no08`, with a sensitivity sweep like `WM_PROB_THRESHOLDS` in `pt2_no07`.
2. **Fisher-Z is applied after averaging in Aim 1.** `pt2_no02` averages raw r per region
   pair and `pt2_no03` then applies `arctanh` to that mean. Aim 2 (`pt2_no05`,
   `pt2_no09`) correctly takes `mean(arctanh(r))`. This violates the invariant in
   `AGENTS.md` §4.2; the fix belongs in `pt2_no02`.
3. **Aim 1's headline test does not adjust for distance.** WM-present pairs are by
   definition tract-connected and therefore systematically closer, and FC decays steeply
   with geodesic distance, so the present-versus-absent effect is confounded. `pt2_no03`
   plots the per-group log-distance fit but does not adjust the t-test.
4. **Region pairs joined by more than one tract are double-weighted** in `pt2_no03`: one
   row per tract carries the same `mean_FC` into the subject average.
5. **The FC ~ log(distance) slope is computed over a different region set** than the rest
   of Aim 2. `distance_targets` in `pt2_no05` and `all_other` in `pt2_no09` are built
   from `np.unique(aal)` and so include cerebellum and brainstem, which `include == 1`
   excludes everywhere else.
6. **DWI → fMRI alignment is assumed, not verified.** `pt1_template_no03` and `no05` are
   reslice-only and rely on the DWI and functional headers sharing a world frame. Against
   TRACULA's own `diff2anatorig.bbr.lta` the header-only mapping is off by a median of
   1.03 mm at real endpoint locations (0.22–1.85 mm across 84 tract-ends) — sub-voxel at
   functional resolution, and corroborated by anatomically sensible AAL assignments. The
   DWI ↔ fMRI leg has not been measured, and `pt1_gm_no01_02` Job 5 rewrites the SPM T1
   header to match the functional data while leaving the DWI header untouched. Verify per
   `README` §4.2 before trusting Aim 2, and note that `pt1_template_no04` compares only
   shape and voxel size, not the affine.
7. **`template/` is untracked but not ignored.** 23 files under `code/template/`
   (`AAL3v1.nii.txt`, `ROI_MNI_V7_1mm.txt`, the AAL3 `*.xml` and `gin_*.m` distribution
   files, NTU-DSI-122 b-vectors) pass `.gitignore` and would be committed by `git add .`,
   contrary to the source-only policy in §7.
8. **`part1/README.md` is out of date** relative to the current scripts (it still
   describes motion-only denoising, no band-pass, and `vector_*` outputs).

Resolved since the last review: the `gm-wm-fc` / `wm-gm-fc` path mismatch in
`pt1_wm_no12`; AAL LUT parsing in `pt1_wm_no07` / `no11`; the nested voxel-pair loop in
`pt2_no02` (vectorised, ~10× faster, verified bit-identical); stale `pt2_no16` /
`pt2_no14` / `vector_*` references in `pt2_no02`; `#` used as a MATLAB comment in
`pt1_template_no02` / `no03`; `BASE_DIRname` in `pt1_template_no04`; the bias-field
helper filename and its `BIAS_FUN` path; and the `AAL3v1_nii.txt` filename in `pt2_no01`.
