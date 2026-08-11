#!/usr/bin/env python3
"""Nested-CV multivariate association of individual mismatch profiles and resilience.

Input features are the RD and out-of-fold structure-function mismatch profiles
made by ``part2_03_build_oof_mismatch_profile.py``.  Sparse feature screening,
covariate adjustment, scaling, PLS fitting, and hyperparameter selection are
all redone inside each training fold.  Consequently, reported predictions are
out-of-sample and resilience information never leaks into profile construction.

This is a discovery analysis.  A significant result identifies a stable
multimodal brain-organisation mode associated with resilience; it is not a
diagnostic or a causal brain-health score.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_PROFILES = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/individual_mismatch/part2_individual_multimodal_profiles.csv"
)
OUTPUT_DIR = (
    "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/"
    "derivatives/pt2_group_results/individual_mismatch/nested_pls_resilience"
)
OUTER_FOLDS = 10
INNER_FOLDS = 5
# First verify the workflow with 25-100 permutations.  Use 1,000 or more only
# for the final result after inspecting QC and out-of-fold predictions.
N_PERMUTATIONS = 100
RANDOM_SEED = 20260807
N_FEATURES_GRID = (6, 12, 20, 30)
N_COMPONENTS_GRID = (1, 2)
# =============================================================================


@dataclass
class PLS2:
    n_components: int

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLS2":
        self.x_mean = x.mean(axis=0)
        self.y_mean = y.mean(axis=0)
        xr = x - self.x_mean
        yr = y - self.y_mean
        ws, ps, cs = [], [], []
        for _ in range(min(self.n_components, x.shape[1], y.shape[1])):
            u, _, _ = np.linalg.svd(xr.T @ yr, full_matrices=False)
            w = u[:, 0]
            t = xr @ w
            denom = float(t @ t)
            if denom <= np.finfo(float).eps:
                break
            c = (yr.T @ t) / denom
            p = (xr.T @ t) / denom
            ws.append(w)
            ps.append(p)
            cs.append(c)
            xr = xr - np.outer(t, p)
            yr = yr - np.outer(t, c)
        if not ws:
            raise RuntimeError("PLS could not extract a non-zero component")
        self.w = np.column_stack(ws)
        self.p = np.column_stack(ps)
        self.c = np.column_stack(cs)
        self.b = self.w @ np.linalg.pinv(self.p.T @ self.w) @ self.c.T
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) @ self.b + self.y_mean

    def x_scores(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) @ self.w


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default=INPUT_PROFILES)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--outer-folds", type=int, default=OUTER_FOLDS)
    p.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    p.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--feature-grid", type=int, nargs="*", default=list(N_FEATURES_GRID))
    p.add_argument("--component-grid", type=int, nargs="*", default=list(N_COMPONENTS_GRID))
    return p.parse_args()


def make_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    if n_folds < 2 or n_folds > n:
        raise ValueError("Number of folds must be between 2 and the sample size")
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    return [fold for fold in np.array_split(idx, n_folds) if len(fold)]


def covariate_matrix(frame: pd.DataFrame, sex_levels: list[str]) -> np.ndarray:
    sex = frame["sex"].astype(str).to_numpy()
    columns = [np.ones(len(frame)), frame["age_c"].to_numpy(dtype=float)]
    for level in sex_levels[1:]:
        columns.append((sex == level).astype(float))
    return np.column_stack(columns)


def residualise(train: np.ndarray, test: np.ndarray, c_train: np.ndarray,
                c_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(c_train, train, rcond=None)[0]
    return train - c_train @ beta, test - c_test @ beta, beta


def standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (train - mean) / sd, (test - mean) / sd, mean, sd


def feature_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maximum absolute univariate relation to either resilience dimension."""
    x0 = x - x.mean(axis=0)
    y0 = y - y.mean(axis=0)
    den_x = np.sqrt((x0 ** 2).sum(axis=0))
    den_y = np.sqrt((y0 ** 2).sum(axis=0))
    den_x[den_x == 0] = np.nan
    den_y[den_y == 0] = np.nan
    corr = (x0.T @ y0) / np.outer(den_x, den_y)
    return np.nanmax(np.abs(corr), axis=1)


def choose_features(x: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    scores = feature_scores(x, y)
    n = min(max(1, n_features), x.shape[1])
    return np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1][:n]


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                n_features: int, n_components: int) -> tuple[np.ndarray, np.ndarray, PLS2]:
    selected = choose_features(train_x, train_y, n_features)
    x_train, x_test, _, _ = standardise(train_x[:, selected], test_x[:, selected])
    y_train, _, y_mean, y_sd = standardise(train_y, train_y)
    model = PLS2(n_components=min(n_components, len(selected), train_y.shape[1])).fit(x_train, y_train)
    prediction = model.predict(x_test) * y_sd + y_mean
    return prediction, selected, model


def inner_choice(x: np.ndarray, y: np.ndarray, cov: np.ndarray, feature_grid: list[int],
                 component_grid: list[int], n_folds: int, seed: int) -> tuple[int, int]:
    folds = make_folds(len(x), min(n_folds, len(x)), seed)
    candidates: list[tuple[float, int, int]] = []
    for n_features in feature_grid:
        for n_components in component_grid:
            losses = []
            for test_idx in folds:
                train_mask = np.ones(len(x), dtype=bool)
                train_mask[test_idx] = False
                xr_train, xr_test, _ = residualise(x[train_mask], x[test_idx], cov[train_mask], cov[test_idx])
                yr_train, yr_test, _ = residualise(y[train_mask], y[test_idx], cov[train_mask], cov[test_idx])
                pred, _, _ = fit_predict(xr_train, yr_train, xr_test, n_features, n_components)
                losses.append(float(np.mean((yr_test - pred) ** 2)))
            candidates.append((float(np.mean(losses)), n_features, n_components))
    # Prefer a smaller model when cross-validated loss is tied to machine precision.
    return min(candidates, key=lambda item: (item[0], item[1], item[2]))[1:]


def run_nested_cv(x: np.ndarray, y: np.ndarray, cov: np.ndarray, outer_folds: int,
                  inner_folds: int, feature_grid: list[int], component_grid: list[int],
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    folds = make_folds(len(x), outer_folds, seed)
    prediction = np.full_like(y, np.nan, dtype=float)
    prediction_residual = np.full_like(y, np.nan, dtype=float)
    actual_residual = np.full_like(y, np.nan, dtype=float)
    selected_counts = np.zeros(x.shape[1], dtype=int)
    choices = []
    for fold_id, test_idx in enumerate(folds):
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[test_idx] = False
        n_features, n_components = inner_choice(
            x[train_mask], y[train_mask], cov[train_mask], feature_grid, component_grid,
            inner_folds, seed + 1000 + fold_id
        )
        xr_train, xr_test, _ = residualise(x[train_mask], x[test_idx], cov[train_mask], cov[test_idx])
        yr_train, yr_test, beta_y = residualise(y[train_mask], y[test_idx], cov[train_mask], cov[test_idx])
        pred_residual, selected, _ = fit_predict(xr_train, yr_train, xr_test, n_features, n_components)
        prediction[test_idx] = pred_residual + cov[test_idx] @ beta_y
        prediction_residual[test_idx] = pred_residual
        actual_residual[test_idx] = yr_test
        selected_counts[selected] += 1
        choices.append({"outer_fold": fold_id, "n_train": int(train_mask.sum()),
                        "n_test": len(test_idx), "n_features": n_features,
                        "n_components": n_components})
    return prediction, prediction_residual, actual_residual, pd.DataFrame(choices), selected_counts


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(values) - rank) * values[index])
        previous = max(previous, value)
        out[index] = previous
    return out.tolist()


def metrics(actual_residual: np.ndarray, predicted_residual: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(["PCA1", "PCA2"]):
        actual = actual_residual[:, i]
        predicted = predicted_residual[:, i]
        r, p = stats.pearsonr(actual, predicted)
        q2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)
        rows.append({"outcome": name, "oof_r": r, "oof_r_squared": r ** 2,
                     "oof_q2": q2, "uncorrected_r_p": p})
    return pd.DataFrame(rows)


def scatter(actual: np.ndarray, predicted: np.ndarray, label: str, output: Path) -> None:
    r, p = stats.pearsonr(actual, predicted)
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.scatter(actual, predicted, alpha=.8, color="#386a9f")
    ax.plot([lo, hi], [lo, hi], color="#aa4542", linewidth=1.5)
    ax.set_xlabel(f"Observed {label} residual")
    ax.set_ylabel(f"Out-of-fold predicted {label} residual")
    ax.set_title(f"Nested-CV brain profile prediction\nr={r:.3f}, p={p:.4g}")
    fig.tight_layout()
    fig.savefig(output / f"part2_nested_pls_{label.lower()}_oof.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(args.input, dtype={"subject": str})
    feature_names = [x for x in profile.columns if x.startswith("rd_profile__") or x.startswith("mismatch_profile__")]
    required = ["subject", "age_c", "sex", "pca1", "pca2"]
    missing = sorted(set(required).difference(profile.columns))
    if missing or not feature_names:
        raise ValueError(f"Missing required columns {missing} or no profile features")
    d = profile.replace([np.inf, -np.inf], np.nan).dropna(subset=required + feature_names).copy()
    if len(d) < max(args.outer_folds * 3, 50):
        raise RuntimeError("Too few complete subjects for nested multivariate analysis.")
    x = d[feature_names].to_numpy(dtype=float)
    y = d[["pca1", "pca2"]].to_numpy(dtype=float)
    sex_levels = sorted(d["sex"].astype(str).unique().tolist())
    cov = covariate_matrix(d, sex_levels)
    feature_grid = sorted({value for value in args.feature_grid if 1 <= value <= x.shape[1]})
    component_grid = sorted({value for value in args.component_grid if 1 <= value <= 2})
    if not feature_grid or not component_grid:
        raise ValueError("Feature/component grids contain no usable values")

    prediction, prediction_residual, residual, choices, selected_counts = run_nested_cv(
        x, y, cov, args.outer_folds, args.inner_folds, feature_grid, component_grid, args.seed
    )
    result = d[["subject", "pca1", "pca2", "age_c", "sex"]].copy()
    result["pca1_residual"] = residual[:, 0]
    result["pca2_residual"] = residual[:, 1]
    result["pca1_predicted_oof"] = prediction[:, 0]
    result["pca2_predicted_oof"] = prediction[:, 1]
    result["pca1_predicted_residual_oof"] = prediction_residual[:, 0]
    result["pca2_predicted_residual_oof"] = prediction_residual[:, 1]
    result.to_csv(output / "part2_nested_pls_oof_predictions.csv", index=False)
    summary = metrics(residual, prediction_residual)

    if args.n_permutations > 0:
        rng = np.random.default_rng(args.seed + 777)
        beta = np.linalg.lstsq(cov, y, rcond=None)[0]
        y_fitted, y_residual = cov @ beta, y - cov @ beta
        null = np.full((args.n_permutations, 2), np.nan)
        for permutation in range(args.n_permutations):
            y_perm = y_fitted + y_residual[rng.permutation(len(y))]
            _, pred_residual_p, residual_p, _, _ = run_nested_cv(
                x, y_perm, cov, args.outer_folds, args.inner_folds,
                feature_grid, component_grid, args.seed + 10000 + permutation
            )
            for outcome in range(2):
                null[permutation, outcome] = stats.pearsonr(
                    residual_p[:, outcome], pred_residual_p[:, outcome]
                )[0]
        null_df = pd.DataFrame(null, columns=["PCA1", "PCA2"])
        null_df.to_csv(output / "part2_nested_pls_permutation_null.csv", index=False)
        permutation_p = []
        for i, outcome in enumerate(["PCA1", "PCA2"]):
            observed = float(summary.loc[summary["outcome"] == outcome, "oof_r"].iloc[0])
            permutation_p.append((1 + int((null_df[outcome].abs() >= abs(observed)).sum())) /
                                 (1 + len(null_df)))
        summary["permutation_p"] = permutation_p
        summary["permutation_p_holm_two_pca_tests"] = holm(permutation_p)

    choices.to_csv(output / "part2_nested_pls_outer_choices.csv", index=False)
    feature_summary = pd.DataFrame({
        "feature": feature_names,
        "outer_fold_selection_count": selected_counts,
        "outer_fold_selection_frequency": selected_counts / args.outer_folds,
    }).sort_values("outer_fold_selection_frequency", ascending=False)
    feature_summary.to_csv(output / "part2_nested_pls_feature_stability.csv", index=False)
    summary.to_csv(output / "part2_nested_pls_prediction_metrics.csv", index=False)

    # A full-data fit is only for describing individual latent brain scores and
    # loadings; all inferential performance remains the OOF result above.
    modal_choice = (choices.groupby(["n_features", "n_components"]).size().reset_index(name="count")
                    .sort_values(["count", "n_features", "n_components"], ascending=[False, True, True]).iloc[0])
    full_x, _, _ = residualise(x, x, cov, cov)
    full_y, _, _ = residualise(y, y, cov, cov)
    final_pred, final_selected, final_model = fit_predict(
        full_x, full_y, full_x, int(modal_choice["n_features"]), int(modal_choice["n_components"])
    )
    selected_x, _, _, _ = standardise(full_x[:, final_selected], full_x[:, final_selected])
    brain_scores = final_model.x_scores(selected_x)
    mode = d[["subject"]].copy()
    for component in range(brain_scores.shape[1]):
        mode[f"brain_mode_{component + 1}_in_sample_descriptive"] = brain_scores[:, component]
    mode.to_csv(output / "part2_individual_multimodal_brain_modes.csv", index=False)
    loading = pd.DataFrame({"feature": np.asarray(feature_names)[final_selected]})
    for component in range(final_model.w.shape[1]):
        loading[f"pls_weight_component_{component + 1}"] = final_model.w[:, component]
    for outcome, name in enumerate(["PCA1", "PCA2"]):
        loading[f"regression_weight_{name}"] = final_model.b[:, outcome]
    loading.to_csv(output / "part2_nested_pls_final_descriptive_loadings.csv", index=False)

    scatter(residual[:, 0], prediction_residual[:, 0], "PCA1", output)
    scatter(residual[:, 1], prediction_residual[:, 1], "PCA2", output)
    pd.DataFrame([{
        "n_subjects": len(d), "n_features": len(feature_names), "n_rd_features": sum(x.startswith("rd_profile__") for x in feature_names),
        "n_mismatch_features": sum(x.startswith("mismatch_profile__") for x in feature_names),
        "outer_folds": args.outer_folds, "inner_folds": args.inner_folds,
        "n_permutations": args.n_permutations,
        "analysis": "nested-CV screened PLS with age/sex residualisation",
    }]).to_csv(output / "part2_nested_pls_run_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
