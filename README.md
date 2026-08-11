# WM–GM FC Connectome Pipeline

Research code for a two-part white-matter/grey-matter functional-connectivity pipeline.

## Active pipeline

The repository uses a canonical, non-versioned layout. The final-to-be pipeline is the
following combination:

| Part | Scope | Active implementation |
| --- | --- | --- |
| Part 1 | Individual-subject preprocessing and individual index generation | `part1/gm`, `part1/wm`, and `part1/template` |
| Part 2 | Integration of individual indices and pattern analysis | `part2/` (`pt2_no01`–`pt2_no10`) |

### Part 1 implementation map

GM setup and preprocessing are retained in `part1/gm`. The intended corrected GM
downstream scripts are:

- `pt1_gm_no03_denoise.m`
- `pt1_gm_no04_detrend_bandpass.m`
- `pt1_gm_no05_masking_binarise.m`
- `pt1_gm_no06_corr_mat_dist.py`
- `pt1_gm_no07_fc_distance_summary.py`

WM and template code are in `part1/wm` and `part1/template`. Superseded variants are
preserved locally in `archives/` and are not tracked by GitHub.

### Part 2 provenance

The source/origin, supplementary, alternate, and exploratory Part 2 scripts are preserved
locally in `archives/` and are not tracked by GitHub. The `part2/pt2_no01`–`pt2_no10`
files are the canonical final-to-be core; `pt2_no08`–`pt2_no10` contain subsequent
refinements.

## Version-control policy

Track source code, configuration, and documentation. Do not commit research data,
participant-derived files, generated matrices/results/figures, logs, local environments,
or credentials. The `archives/` directory is deliberately retained locally and ignored.

Before a script is shared externally, replace machine- or institution-specific paths
with documented configuration variables.
