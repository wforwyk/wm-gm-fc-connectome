# WM–GM FC Connectome Pipeline

Research code for a two-part white-matter/grey-matter functional-connectivity pipeline.

## Active pipeline

The repository uses a canonical, non-versioned layout. The final-to-be pipeline is the
following combination:

| Part | Scope | Active implementation |
| --- | --- | --- |
| Part 1 | Individual-subject preprocessing and individual index generation | `part1/gm`, `part1/wm`, and `part1/template` |
| Part 2 | Integration of individual indices and pattern analysis | `part2/` (`pt2_no1`–`pt2_no10`) |

### Part 1 implementation map

GM setup and preprocessing are retained in `part1/gm`. The intended corrected GM
downstream scripts are:

- `gm_no3_denoise_v4.m`
- `gm_no4_detrend_bandpass_v3.m`
- `gm_no5_masking_binarise_v3.m`
- `gm_no6_corr_mat_dist_v4.py`
- `gm_no7_fc_distance_summary_v3.py`

WM and template code are in `part1/wm` and `part1/template`. The other numbered variants
remain tracked as development history and should not be deleted until the final pipeline
is validated.

### Part 2 provenance

`part2/development_snapshot` preserves the source/origin of the Part 2 work, including
supplementary, alternate, and exploratory scripts. The `part2/pt2_no1`–`pt2_no10` files
are the canonical final-to-be core; `pt2_no8`–`pt2_no10` contain subsequent refinements.

## Version-control policy

Track source code, configuration, and documentation. Do not commit research data,
participant-derived files, generated matrices/results/figures, logs, local environments,
or credentials. The `archives/` directory is deliberately retained locally and ignored.

Before a script is shared externally, replace machine- or institution-specific paths
with documented configuration variables.
