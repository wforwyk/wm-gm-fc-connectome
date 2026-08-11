#!/usr/bin/env python3
"""Primary resilience analysis for the Part 2 subject-by-tract table.

Primary estimand
-----------------
Does resilience alter the association between *relative tract integrity*
(-within-subject z(RD)) and distance-adjusted tract FC?  The main model uses
subject and tract fixed effects with subject-clustered standard errors:

    FC_z ~ log(distance) + integrity + integrity:PCA1 + integrity:PCA2
           + tract fixed effects + subject fixed effects

Age and sex can confound the *within-subject integrity slope*, so their
interactions with integrity are included.  Their main effects are absorbed by
the subject fixed effects.  A random-intercept/random-integrity-slope mixed
model is emitted as a sensitivity analysis when it converges.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, stats
from patsy import dmatrix
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# =============================================================================
# USER SETTINGS
# Edit this block for the server.  Command-line arguments still override these
# values, so scripts can be run either directly or from a batch command.
# =============================================================================
INPUT_TABLE = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/final_resilience/part2_subject_tract_primary.csv"
)
OUTPUT_DIR = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/final_resilience/model"
)
N_PERMUTATIONS = 5000
RANDOM_SEED = 20260807
RUN_MIXEDLM_SENSITIVITY = True
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", default=INPUT_TABLE,
                        help="part2_subject_tract_primary.csv from script 01")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help="0 skips subject-label permutation inference")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-mixedlm", action="store_true",
                        default=not RUN_MIXEDLM_SENSITIVITY)
    return parser.parse_args()


PRIMARY_TERMS = ["rd_integrity_within:pca1_z", "rd_integrity_within:pca2_z"]


def holm_adjust(values: pd.Series) -> pd.Series:
    """Holm-adjust only finite values while retaining their original rows."""
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if valid.any():
        adjusted.loc[valid] = multipletests(values.loc[valid], method="holm")[1]
    return adjusted


def write_model(fit, output: Path, name: str) -> pd.DataFrame:
    frame = pd.DataFrame({
        "term": fit.params.index,
        "estimate": fit.params.values,
        "se": fit.bse.values,
        "statistic": fit.tvalues.values,
        "p": fit.pvalues.values,
    })
    ci = np.asarray(fit.conf_int())
    frame["ci95_lo"] = ci[:, 0]
    frame["ci95_hi"] = ci[:, 1]
    frame.to_csv(output / f"{name}_coefficients.csv", index=False)
    with open(output / f"{name}_summary.txt", "w") as handle:
        handle.write(fit.summary().as_text())
    return frame


def fit_primary(data: pd.DataFrame):
    formula = (
        "fc_z ~ log_mean_dist + rd_integrity_within "
        "+ rd_integrity_within:pca1_z + rd_integrity_within:pca2_z "
        "+ rd_integrity_within:age_c + rd_integrity_within:C(sex) "
        "+ C(clean_tract) + C(subject)"
    )
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["subject"]}
    ), formula


def residualize_with_nuisance(values: np.ndarray, q_basis: np.ndarray) -> np.ndarray:
    return values - q_basis @ (q_basis.T @ values)


def nuisance_q_basis(data: pd.DataFrame) -> np.ndarray:
    """Numerically stable orthonormal basis for all non-PCA interaction terms."""
    nuisance_formula = (
        "1 + log_mean_dist + rd_integrity_within "
        "+ rd_integrity_within:age_c + rd_integrity_within:C(sex) "
        "+ C(clean_tract) + C(subject)"
    )
    design = np.asarray(dmatrix(nuisance_formula, data=data, return_type="dataframe"), dtype=float)
    q, r, _ = linalg.qr(design, mode="economic", pivoting=True, check_finite=True)
    diagonal = np.abs(np.diag(r))
    tolerance = np.finfo(float).eps * max(design.shape) * (diagonal.max() if len(diagonal) else 1.0)
    rank = int((diagonal > tolerance).sum())
    return q[:, :rank]


def permutation_test(data: pd.DataFrame, observed: pd.Series,
                     n_permutations: int, seed: int) -> pd.DataFrame:
    """Permute PCA residual pairs after one stable subject/tract residualization.

    Re-fitting hundreds of subject fixed effects for every permutation is both
    inefficient and numerically unstable.  Frisch-Waugh-Lovell residualization
    yields the identical OLS interaction estimate while keeping each
    permutation a two-column regression.
    """
    rng = np.random.default_rng(seed)
    subject_scores = data[["subject", "pca1_z", "pca2_z", "age_c", "sex"]].drop_duplicates("subject")
    ids = subject_scores["subject"].to_numpy()
    scores = subject_scores[["pca1_z", "pca2_z"]].to_numpy()
    # Preserve any age/sex association of the two PCA scores.  Permuting raw
    # scores would spuriously remove that association despite its inclusion in
    # the primary model as a moderator of the integrity slope.
    design = pd.get_dummies(subject_scores[["age_c", "sex"]], columns=["sex"], drop_first=True)
    design = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
    fitted = design @ np.linalg.lstsq(design, scores, rcond=None)[0]
    residuals = scores - fitted
    subject_lookup = {subject: i for i, subject in enumerate(ids)}
    row_subject = np.asarray([subject_lookup[x] for x in data["subject"]], dtype=int)
    rd = data["rd_integrity_within"].to_numpy(dtype=float)
    q_basis = nuisance_q_basis(data)
    y_residual = residualize_with_nuisance(data["fc_z"].to_numpy(dtype=float), q_basis)
    values = np.full((n_permutations, len(PRIMARY_TERMS)), np.nan)
    for i in range(n_permutations):
        permuted_scores = fitted + residuals[rng.permutation(len(ids))]
        interaction = rd[:, None] * permuted_scores[row_subject]
        interaction_residual = residualize_with_nuisance(interaction, q_basis)
        values[i] = np.linalg.lstsq(interaction_residual, y_residual, rcond=None)[0]
    result = pd.DataFrame(values, columns=PRIMARY_TERMS)
    rows = []
    for term in PRIMARY_TERMS:
        null = result[term].dropna().abs()
        estimate = float(observed.loc[term])
        rows.append({
            "term": term,
            "observed_estimate": estimate,
            "n_permutations": len(null),
            "two_sided_permutation_p": (1 + int((null >= abs(estimate)).sum())) / (1 + len(null)),
        })
    return result, pd.DataFrame(rows)


def individual_phenotypes(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    """A descriptive subject-level coupling slope after distance/tract adjustment."""
    base = smf.ols("fc_z ~ log_mean_dist + C(clean_tract) + C(subject)", data=data).fit()
    d = data.copy()
    d["fc_distance_tract_residual"] = base.resid
    rows = []
    for subject, group in d.groupby("subject"):
        if group["rd_integrity_within"].nunique() < 3 or len(group) < 8:
            continue
        fit = smf.ols("fc_distance_tract_residual ~ rd_integrity_within", data=group).fit()
        rows.append({
            "subject": subject,
            "n_tracts": len(group),
            "coupling_slope": fit.params["rd_integrity_within"],
            "coupling_slope_se": fit.bse["rd_integrity_within"],
            "coupling_slope_r2": fit.rsquared,
            "pca1_z": group["pca1_z"].iloc[0],
            "pca2_z": group["pca2_z"].iloc[0],
            "age_c": group["age_c"].iloc[0],
            "sex": group["sex"].iloc[0],
        })
    subject = pd.DataFrame(rows)
    subject.to_csv(output / "part2_individual_coupling_phenotypes.csv", index=False)
    if len(subject) >= 10:
        fit = smf.ols("coupling_slope ~ pca1_z + pca2_z + age_c + C(sex)", data=subject).fit(cov_type="HC3")
        write_model(fit, output, "part2_individual_phenotype_association")
    return subject


def scatter_plot(subject: pd.DataFrame, x: str, label: str, output: Path) -> None:
    if subject.empty:
        return
    d = subject[[x, "coupling_slope"]].dropna()
    if len(d) < 3:
        return
    fit = stats.linregress(d[x], d["coupling_slope"])
    xx = np.linspace(d[x].min(), d[x].max(), 100)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(d[x], d["coupling_slope"], color="#3568a8", alpha=.8)
    ax.plot(xx, fit.intercept + fit.slope * xx, color="#b4413e", linewidth=2)
    ax.axhline(0, color="0.6", linewidth=.8)
    ax.set_xlabel(f"{label} (z)")
    ax.set_ylabel("Individual RD–FC coupling slope")
    ax.set_title(f"Coupling phenotype vs {label}\nr={fit.rvalue:.3f}, p={fit.pvalue:.4g}")
    fig.tight_layout()
    fig.savefig(output / f"part2_coupling_vs_{x}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input, dtype={"subject": str})
    needed = ["fc_z", "log_mean_dist", "rd_integrity_within", "pca1_z", "pca2_z",
              "age_c", "sex", "clean_tract", "subject"]
    missing = sorted(set(needed).difference(data.columns))
    if missing:
        raise ValueError(f"Input table is missing: {missing}")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    if data["subject"].nunique() < 10:
        raise RuntimeError("Fewer than 10 eligible subjects; do not run a resilience association model.")

    primary, _ = fit_primary(data)
    coeff = write_model(primary, output, "part2_primary_subject_tract_model")
    key = coeff.loc[coeff["term"].isin(PRIMARY_TERMS)].copy()
    key["inference"] = "subject-clustered standard errors; subject and tract fixed effects"
    key["p_holm_two_pca_tests"] = holm_adjust(key["p"])

    if args.n_permutations > 0:
        null, permutation = permutation_test(
            data, primary.params, args.n_permutations, args.seed
        )
        permutation["permutation_p_holm_two_pca_tests"] = holm_adjust(
            permutation["two_sided_permutation_p"]
        )
        null.to_csv(output / "part2_primary_permutation_null.csv", index=False)
        permutation.to_csv(output / "part2_primary_permutation_tests.csv", index=False)
        key = key.merge(
            permutation[["term", "two_sided_permutation_p", "permutation_p_holm_two_pca_tests"]],
            on="term", how="left", validate="one_to_one"
        )
    key.to_csv(output / "part2_primary_resilience_tests.csv", index=False)

    if not args.skip_mixedlm:
        mixed_formula = (
            "fc_z ~ log_mean_dist + rd_integrity_within * pca1_z "
            "+ rd_integrity_within * pca2_z + rd_integrity_within:age_c "
            "+ rd_integrity_within:C(sex) + C(clean_tract) + age_c + C(sex)"
        )
        try:
            mixed = smf.mixedlm(
                mixed_formula, data=data, groups=data["subject"],
                re_formula="1 + rd_integrity_within"
            ).fit(reml=False, method="lbfgs", maxiter=500)
            write_model(mixed, output, "part2_mixedlm_sensitivity")
        except Exception as exc:  # a non-convergent sensitivity model must not halt the primary analysis
            (output / "part2_mixedlm_sensitivity_error.txt").write_text(str(exc) + "\n")

    phenotype = individual_phenotypes(data, output)
    scatter_plot(phenotype, "pca1_z", "PCA1", output)
    scatter_plot(phenotype, "pca2_z", "PCA2", output)
    pd.DataFrame([{
        "n_subjects": data["subject"].nunique(),
        "n_subject_tract_rows": len(data),
        "n_tracts": data["clean_tract"].nunique(),
        "primary_outcome": "fc_z (mean Fisher-z FC)",
        "primary_tests": "rd_integrity_within:PCA1 and rd_integrity_within:PCA2",
    }]).to_csv(output / "part2_model_sample_summary.csv", index=False)
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
