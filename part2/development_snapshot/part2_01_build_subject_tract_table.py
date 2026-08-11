#!/usr/bin/env python3
"""Build the final Part 2 subject-by-tract analysis table.

The completed Aim 1 pooled file contains one or more AAL-pair observations per
subject and tract.  This script retains only tract-mapped (``wm_present``)
observations, creates exactly one row per subject and tract, joins the
demographic/PCA file, and makes QC outputs.  The resulting table is the only
input required by ``part2_02_resilience_coupling.py``.

The primary outcome is ``mean_FC_z``: the mean Fisher-z transformed FC.  Raw
``mean_FC`` is kept only as a descriptive value.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# USER SETTINGS
# Edit this block for the server.  Command-line arguments still override these
# values, so the script can also be reused without editing it.
# =============================================================================
BATCH_DIR = "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch"
AIM1_POOLED = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/no3_summary_stats_v9/pt2_no3_pooled.csv"
)
DEMOGRAPHICS = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "subjects/subjects_demographic.xlsx"
)
OUTPUT_DIR = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/final_resilience"
)
# Empty list = all subjects that are present in Aim 1 and the demographic file.
SUBJECT_LIST: list[str] = []
# Empty list = retain every tract already marked wm_present in the Aim 1 file.
EXCLUDE_TRACTS: list[str] = []
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--aim1-pooled", default=AIM1_POOLED,
                        help="pt2_no3_pooled.csv from completed Aim 1")
    parser.add_argument("--demographics", default=DEMOGRAPHICS,
                        help="CSV or XLSX with Subject, Sex, Age, PCA1 and PCA2")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--batch-dir", default=BATCH_DIR,
                        help="derivatives/batch directory; use '' to skip RD_Avg cross-check")
    parser.add_argument("--subjects", nargs="*", default=SUBJECT_LIST,
                        help="Optional subject IDs; default uses all eligible subjects")
    parser.add_argument("--exclude-tracts", nargs="*", default=EXCLUDE_TRACTS,
                        help="Optional cleaned tract names to exclude")
    parser.add_argument("--subject-col", default="Subject")
    parser.add_argument("--sex-col", default="Sex")
    parser.add_argument("--age-col", default="Age")
    parser.add_argument("--pca1-col", default="PCA1")
    parser.add_argument("--pca2-col", default="PCA2")
    parser.add_argument("--wm-group-col", default="wm_group")
    parser.add_argument("--wm-present-label", default="wm_present")
    parser.add_argument("--tract-col", default="tract_name")
    parser.add_argument("--fc-z-col", default="mean_FC_z")
    parser.add_argument("--fc-r-col", default="mean_FC")
    parser.add_argument("--distance-col", default="mean_dist_mm")
    parser.add_argument("--rd-col", default="mean_RD")
    parser.add_argument("--pair-count-col", default="n_voxel_pairs")
    parser.add_argument("--tract-suffix", default="_avg16_syn_bbr")
    parser.add_argument("--pathstats-rd-col", default="RD_Avg")
    return parser.parse_args()


def canonical_subject(value: object) -> str:
    """Represent spreadsheet IDs consistently, including Excel's 1001.0 form."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if keep.any():
        return float(np.average(x[keep], weights=w[keep]))
    return float(np.nanmean(x))


def zscore_within_subject(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / sd


def clean_tract(values: pd.Series, suffix: str) -> pd.Series:
    return values.astype(str).str.strip().str.replace(suffix, "", regex=False)


def verify_pathstats(table: pd.DataFrame, batch_dir: str, rd_col: str,
                     tract_suffix: str, rd_name: str) -> pd.DataFrame:
    """Read per-subject pathstats only for a transparent RD consistency QC."""
    rows: list[dict[str, object]] = []
    batch = Path(batch_dir)
    for subject in sorted(table["subject"].unique()):
        path = batch / subject / "wm" / "derived" / "pathstats" / f"{subject}_pathstats.csv"
        if not path.exists():
            rows.append({"subject": subject, "status": "missing_pathstats", "message": str(path)})
            continue
        ps = pd.read_csv(path)
        if "tract" not in ps or rd_name not in ps:
            rows.append({"subject": subject, "status": "missing_required_columns",
                         "message": f"need tract and {rd_name}"})
            continue
        ps = ps[["tract", rd_name]].copy()
        ps["clean_tract"] = clean_tract(ps["tract"], tract_suffix)
        source = table.loc[table["subject"] == subject,
                           ["subject", "clean_tract", rd_col]].copy()
        merged = source.merge(ps[["clean_tract", rd_name]], on="clean_tract", how="left")
        for item in merged.itertuples(index=False):
            source_rd = getattr(item, rd_col)
            pathstats_rd = getattr(item, rd_name)
            rows.append({
                "subject": subject,
                "clean_tract": item.clean_tract,
                "mean_rd_from_aim1": source_rd,
                "rd_avg_from_pathstats": pathstats_rd,
                "difference": source_rd - pathstats_rd
                if np.isfinite(source_rd) and np.isfinite(pathstats_rd) else np.nan,
                "status": "ok" if np.isfinite(pathstats_rd) else "tract_not_in_pathstats",
                "message": "",
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(args.aim1_pooled)
    required = {
        "subject", args.wm_group_col, args.tract_col, args.fc_z_col,
        args.fc_r_col, args.distance_col, args.rd_col, args.pair_count_col,
    }
    missing = sorted(required.difference(pooled.columns))
    if missing:
        raise ValueError(f"Aim 1 pooled file is missing columns: {missing}")

    pooled = pooled.copy()
    pooled["subject"] = pooled["subject"].map(canonical_subject)
    pooled["clean_tract"] = clean_tract(pooled[args.tract_col], args.tract_suffix)
    present = pooled[args.wm_group_col].astype(str).str.strip().eq(args.wm_present_label)
    table = pooled.loc[present & pooled["clean_tract"].ne("")].copy()
    if args.subjects:
        requested = {canonical_subject(x) for x in args.subjects}
        table = table.loc[table["subject"].isin(requested)].copy()
    if args.exclude_tracts:
        excluded = {str(x).strip() for x in args.exclude_tracts}
        table = table.loc[~table["clean_tract"].isin(excluded)].copy()
    table = table.replace([np.inf, -np.inf], np.nan)
    for col in [args.fc_z_col, args.fc_r_col, args.distance_col, args.rd_col, args.pair_count_col]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(subset=[args.fc_z_col, args.distance_col, args.rd_col])
    table = table.loc[table[args.distance_col] > 0].copy()
    if table.empty:
        raise RuntimeError("No usable wm_present rows remained after QC.")

    # A tract can occasionally appear more than once after AAL-pair mapping.
    # Aggregate it deliberately, retaining the source-row count for audit.
    records = []
    for (subject, tract), group in table.groupby(["subject", "clean_tract"], sort=True):
        records.append({
            "subject": subject,
            "clean_tract": tract,
            "fc_z": weighted_mean(group[args.fc_z_col], group[args.pair_count_col]),
            "fc_r_descriptive": weighted_mean(group[args.fc_r_col], group[args.pair_count_col]),
            "mean_dist_mm": weighted_mean(group[args.distance_col], group[args.pair_count_col]),
            "mean_rd": weighted_mean(group[args.rd_col], group[args.pair_count_col]),
            "n_voxel_pairs": int(np.nansum(group[args.pair_count_col])),
            "n_aim1_rows": int(len(group)),
            "tract_name_aim1": " | ".join(sorted(group[args.tract_col].dropna().astype(str).unique())),
        })
    st = pd.DataFrame(records)
    st["log_mean_dist"] = np.log(st["mean_dist_mm"])
    # Higher value means lower RD, hence relatively more preserved microstructure.
    st["rd_integrity_within"] = -st.groupby("subject")["mean_rd"].transform(zscore_within_subject)
    st["mean_rd_subject"] = st.groupby("subject")["mean_rd"].transform("mean")

    demographics = read_table(args.demographics).copy()
    demographic_cols = [args.subject_col, args.sex_col, args.age_col, args.pca1_col, args.pca2_col]
    missing_demo = sorted(set(demographic_cols).difference(demographics.columns))
    if missing_demo:
        raise ValueError(f"Demographic file is missing columns: {missing_demo}")
    demo = demographics[demographic_cols].copy()
    demo["subject"] = demo[args.subject_col].map(canonical_subject)
    demo = demo.drop(columns=[args.subject_col]).rename(columns={
        args.sex_col: "sex", args.age_col: "age", args.pca1_col: "pca1", args.pca2_col: "pca2",
    })
    demo = demo.drop_duplicates("subject", keep=False)
    for col in ["age", "pca1", "pca2"]:
        demo[col] = pd.to_numeric(demo[col], errors="coerce")
    st = st.merge(demo, on="subject", how="left", validate="many_to_one")
    st["age_c"] = st["age"] - st["age"].mean()
    for col in ["pca1", "pca2"]:
        sd = st[["subject", col]].drop_duplicates()[col].std(ddof=1)
        mean = st[["subject", col]].drop_duplicates()[col].mean()
        st[f"{col}_z"] = (st[col] - mean) / sd if np.isfinite(sd) and sd > 0 else np.nan

    required_analysis = ["rd_integrity_within", "pca1_z", "pca2_z", "age_c", "sex"]
    st["eligible_primary_model"] = st[required_analysis].notna().all(axis=1)
    subject_qc = (st.groupby("subject", as_index=False)
                  .agg(n_tracts=("clean_tract", "nunique"),
                       n_rows=("clean_tract", "size"),
                       eligible_primary_model=("eligible_primary_model", "all"),
                       mean_rd=("mean_rd", "mean"),
                       mean_fc_z=("fc_z", "mean"),
                       mean_distance_mm=("mean_dist_mm", "mean")))

    qc = pd.DataFrame([{
        "n_aim1_rows_total": len(pooled),
        "n_aim1_wm_present_rows": int(present.sum()),
        "n_subject_tract_rows": len(st),
        "n_subjects": st["subject"].nunique(),
        "n_primary_eligible_subjects": int(subject_qc["eligible_primary_model"].sum()),
        "fc_outcome": args.fc_z_col,
        "raw_fc_descriptive": args.fc_r_col,
        "rd_variable": args.rd_col,
        "distance_variable": args.distance_col,
    }])

    st.sort_values(["subject", "clean_tract"]).to_csv(out / "part2_subject_tract_primary.csv", index=False)
    subject_qc.sort_values("subject").to_csv(out / "part2_subject_qc.csv", index=False)
    qc.to_csv(out / "part2_build_qc_summary.csv", index=False)
    table.sort_values(["subject", "clean_tract"]).to_csv(out / "part2_aim1_wm_present_source_rows.csv", index=False)
    with open(out / "part2_analysis_definition.json", "w") as handle:
        json.dump({
            "analysis_unit": "one subject by one tract",
            "outcome": "fc_z = pair-count weighted mean of mean_FC_z",
            "microstructure": "rd_integrity_within = - within-subject z(mean_RD)",
            "distance": "log(mean_dist_mm)",
            "resilience_variables": ["pca1_z", "pca2_z"],
            "source": os.path.abspath(args.aim1_pooled),
        }, handle, indent=2)

    if args.batch_dir:
        rd_qc = verify_pathstats(st, args.batch_dir, "mean_rd", args.tract_suffix,
                                 args.pathstats_rd_col)
        rd_qc.to_csv(out / "part2_rd_pathstats_crosscheck.csv", index=False)

    print(qc.to_string(index=False))
    print(f"Wrote final analysis table to {out / 'part2_subject_tract_primary.csv'}")


if __name__ == "__main__":
    main()
