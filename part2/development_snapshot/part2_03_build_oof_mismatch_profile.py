#!/usr/bin/env python3
"""Create individual out-of-fold tract structure-function mismatch profiles.

For each held-out subject, this script learns the expected *within-subject FC
profile* from all other subjects.  Expected FC is conditioned on tract identity,
linear/quadratic log-geodesic distance, age, sex, and relative tract RD.  The
held-out residual is therefore a subject-specific, distance- and RD-adjusted
FC deviation (``sf_mismatch_oof``), not an in-sample fitted residual.

No resilience variable is used here.  This protects the individual brain
phenotype from outcome leakage before the subsequent brain--resilience model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_TABLE = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/final_resilience/part2_subject_tract_primary.csv"
)
OUTPUT_DIR = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/individual_mismatch"
)
N_FOLDS = 10
RANDOM_SEED = 20260807
# A participant must have this many usable tracts to receive a profile.
MIN_TRACTS_PER_SUBJECT = 30
# A tract must be available in this proportion of retained participants to be
# part of the common wide feature matrix for the next script.
MIN_TRACT_COVERAGE = 0.95
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=INPUT_TABLE)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--min-tracts", type=int, default=MIN_TRACTS_PER_SUBJECT)
    p.add_argument("--min-tract-coverage", type=float, default=MIN_TRACT_COVERAGE)
    return p.parse_args()


def subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds < 2 or n_folds > len(subjects):
        raise ValueError("n_folds must be between 2 and the number of subjects")
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(subjects, dtype=str).copy()
    rng.shuffle(shuffled)
    return [x for x in np.array_split(shuffled, n_folds) if len(x)]


def model_formula(include_rd: bool) -> str:
    base = (
        "fc_z_subject_centered ~ log_mean_dist + log_mean_dist_sq "
        "+ age_c + C(sex) + C(clean_tract)"
    )
    return base + " + rd_integrity_within" if include_rd else base


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, dtype={"subject": str})
    required = [
        "subject", "clean_tract", "fc_z", "log_mean_dist", "rd_integrity_within",
        "mean_rd", "age_c", "sex", "pca1", "pca2", "pca1_z", "pca2_z",
    ]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    tract_counts = data.groupby("subject")["clean_tract"].nunique()
    kept_subjects = tract_counts.loc[tract_counts >= args.min_tracts].index.astype(str)
    data = data.loc[data["subject"].isin(kept_subjects)].copy()
    if data["subject"].nunique() < args.n_folds:
        raise RuntimeError("Too few retained subjects for the requested number of folds.")

    # Remove person-wide FC level: the profile then reflects organisation across
    # tracts, rather than a global FC amplitude difference.
    data["fc_z_subject_centered"] = data["fc_z"] - data.groupby("subject")["fc_z"].transform("mean")
    data["log_mean_dist_sq"] = data["log_mean_dist"] ** 2
    data["oof_fold"] = -1
    data["fc_distance_expected_oof"] = np.nan
    data["fc_distance_residual_oof"] = np.nan
    data["fc_rd_expected_oof"] = np.nan
    data["sf_mismatch_oof"] = np.nan
    folds = subject_folds(data["subject"].drop_duplicates().to_numpy(), args.n_folds, args.seed)
    fold_rows = []

    for fold_id, heldout in enumerate(folds):
        train = data.loc[~data["subject"].isin(heldout)].copy()
        test_mask = data["subject"].isin(heldout)
        test = data.loc[test_mask].copy()
        missing_tracts = set(test["clean_tract"]) - set(train["clean_tract"])
        if missing_tracts:
            raise RuntimeError(f"Fold {fold_id}: held-out-only tracts: {sorted(missing_tracts)}")
        distance_fit = smf.ols(model_formula(include_rd=False), data=train).fit()
        rd_fit = smf.ols(model_formula(include_rd=True), data=train).fit()
        pred_distance = np.asarray(distance_fit.predict(test), dtype=float)
        pred_rd = np.asarray(rd_fit.predict(test), dtype=float)
        observed = test["fc_z_subject_centered"].to_numpy(dtype=float)
        data.loc[test_mask, "oof_fold"] = fold_id
        data.loc[test_mask, "fc_distance_expected_oof"] = pred_distance
        data.loc[test_mask, "fc_distance_residual_oof"] = observed - pred_distance
        data.loc[test_mask, "fc_rd_expected_oof"] = pred_rd
        data.loc[test_mask, "sf_mismatch_oof"] = observed - pred_rd
        fold_rows.append({
            "fold": fold_id,
            "n_train_subjects": train["subject"].nunique(),
            "n_test_subjects": test["subject"].nunique(),
            "n_test_rows": len(test),
            "distance_model_rmse": float(np.sqrt(np.mean((observed - pred_distance) ** 2))),
            "rd_model_rmse": float(np.sqrt(np.mean((observed - pred_rd) ** 2))),
            "distance_model_r2_test": float(1 - np.sum((observed - pred_distance) ** 2) /
                                              np.sum((observed - observed.mean()) ** 2)),
            "rd_model_r2_test": float(1 - np.sum((observed - pred_rd) ** 2) /
                                        np.sum((observed - observed.mean()) ** 2)),
        })

    if data["sf_mismatch_oof"].isna().any():
        raise RuntimeError("Some rows did not receive an out-of-fold mismatch value.")
    data["abs_sf_mismatch_oof"] = data["sf_mismatch_oof"].abs()
    data.to_csv(output / "part2_oof_tract_mismatch_long.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "part2_oof_model_fold_qc.csv", index=False)

    coverage = data.pivot_table(index="subject", columns="clean_tract", values="sf_mismatch_oof",
                                aggfunc="first").notna().mean(axis=0)
    retained_tracts = coverage.loc[coverage >= args.min_tract_coverage].index.tolist()
    rd_wide = data.pivot_table(index="subject", columns="clean_tract", values="rd_integrity_within",
                               aggfunc="first").reindex(columns=retained_tracts)
    mismatch_wide = data.pivot_table(index="subject", columns="clean_tract", values="sf_mismatch_oof",
                                     aggfunc="first").reindex(columns=retained_tracts)
    rd_wide.columns = [f"rd_profile__{x}" for x in rd_wide.columns]
    mismatch_wide.columns = [f"mismatch_profile__{x}" for x in mismatch_wide.columns]
    demographics = (data.groupby("subject", as_index=True)
                    .agg(age=("age", "first"), sex=("sex", "first"), age_c=("age_c", "first"),
                         pca1=("pca1", "first"), pca2=("pca2", "first"),
                         pca1_z=("pca1_z", "first"), pca2_z=("pca2_z", "first")))
    summaries = (data.groupby("subject", as_index=True)
                 .agg(n_tracts=("clean_tract", "nunique"),
                      mean_abs_sf_mismatch=("abs_sf_mismatch_oof", "mean"),
                      sd_sf_mismatch=("sf_mismatch_oof", "std"),
                      mean_sf_mismatch=("sf_mismatch_oof", "mean"),
                      mean_rd=("mean_rd", "mean"), mean_fc_z=("fc_z", "mean")))
    wide = demographics.join(summaries).join(rd_wide).join(mismatch_wide).reset_index()
    wide.to_csv(output / "part2_individual_multimodal_profiles.csv", index=False)
    pd.DataFrame({"clean_tract": coverage.index, "subject_coverage": coverage.values,
                  "retained_for_wide_profile": coverage.index.isin(retained_tracts)}).to_csv(
        output / "part2_profile_tract_coverage.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.hist(summaries["mean_abs_sf_mismatch"].dropna(), bins=28, color="#3b6ca8", edgecolor="white")
    ax.set_xlabel("Mean absolute out-of-fold structure–function mismatch")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Individual mismatch-profile burden")
    fig.tight_layout()
    fig.savefig(output / "part2_mismatch_burden_distribution.png", dpi=220)
    plt.close(fig)

    pd.DataFrame([{
        "n_subjects": data["subject"].nunique(),
        "n_subject_tract_rows": len(data),
        "n_retained_tracts_for_profile": len(retained_tracts),
        "n_folds": len(folds),
        "definition": "OOF FC residual after distance, tract, age, sex, and relative RD adjustment",
    }]).to_csv(output / "part2_oof_mismatch_summary.csv", index=False)
    print(pd.read_csv(output / "part2_oof_mismatch_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
