#!/usr/bin/env python3
"""
pt2_no13_tract_network_stats.py
================================
Test participant-level tract-system and network effects.

All four tract classes (commissural, projection,
association, other = bilateral fornix) are part of the main analysis.

Tract-system tests
------------------
1. Per-class endpoint interaction versus zero for all four classes
   (BH-FDR across the four tests; family "tract_system_each").
2. Commissural-minus-association contrast.

Secondary canonical-network contrasts
-------------------------------------
1. Within-network minus between-network effects across cortical units of all
   four classes (Yeo7 has no thalamic labels, so projection tracts contribute
   only cortico-cortical units).
2. Tract-system x network-relation interaction (commissural vs association).
3. Within-minus-between effects separately in each class with support
   (FDR-corrected as one secondary family).

The main network mapping assigns each cortical AAL3 ROI to the Yeo7 network
with maximum SST overlap.  The same tests are repeated as a
sensitivity analysis after requiring a majority Yeo assignment at both ends.
Tests use participant-level effects or paired contrasts, a one-sample t-test,
participant sign-flip permutation p value, and bootstrap confidence interval.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pt2_no12_tract_network_prepare import system_effects, validate_tract_classification


from cohort_settings import SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, select_subjects, select_files, filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
INPUT_DIR = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no12_tract_network_prepare"
)
TRACT_EFFECTS_PATH = INPUT_DIR / "aim2_tract_effects.csv"
PARTICIPANT_SYSTEM_PATH = INPUT_DIR / "aim2_participant_system_effects.csv"
OUTPUT_DIR = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no13_tract_network_stats"
)
N_SIGN_FLIP = 100_000
N_BOOTSTRAP = 20_000
RANDOM_SEED = 20260904
# =============================================================================


PLANNED_PAIR = ["commissural", "association"]
ALL_SYSTEMS = ["commissural", "projection", "association", "other"]


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return adjusted
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    corrected = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.minimum(corrected, 1.0)
    adjusted[valid] = restored
    return adjusted


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator) -> float:
    """Two-sided Monte-Carlo sign-flip test of the participant mean."""
    observed = abs(float(values.mean()))
    if len(values) < 2:
        return np.nan
    exceed = 0
    remaining = N_SIGN_FLIP
    chunk = min(10_000, N_SIGN_FLIP)
    while remaining:
        size = min(chunk, remaining)
        signs = rng.integers(0, 2, size=(size, len(values)), dtype=np.int8) * 2 - 1
        permuted_means = (signs * values).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted_means) >= observed))
        remaining -= size
    return (exceed + 1) / (N_SIGN_FLIP + 1)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).astype(float))


def paired_test(
    frame: pd.DataFrame,
    index_columns: list[str],
    condition_column: str,
    left: str,
    right: str,
    label: str,
    family: str,
    mapping_variant: str,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame]:
    pivot = frame.pivot_table(
        index=index_columns, columns=condition_column, values="aim2_interaction", aggfunc="mean"
    )
    if left not in pivot.columns or right not in pivot.columns:
        delta = np.array([], dtype=float)
        subject_contrast = pd.DataFrame(columns=[*index_columns, "left", "right", "contrast"])
    else:
        paired = pivot.loc[:, [left, right]].dropna()
        delta = (paired[left] - paired[right]).to_numpy(dtype=float)
        subject_contrast = paired.reset_index().rename(columns={left: "left", right: "right"})
        subject_contrast["contrast"] = subject_contrast["left"] - subject_contrast["right"]
    result = {
        "test": label,
        "family": family,
        "mapping_variant": mapping_variant,
        "left_condition": left,
        "right_condition": right,
        "n_subjects": int(len(delta)),
        "mean_contrast": float(np.mean(delta)) if len(delta) else np.nan,
        "se": float(np.std(delta, ddof=1) / np.sqrt(len(delta))) if len(delta) >= 2 else np.nan,
    }
    if len(delta) >= 2:
        t, p = stats.ttest_1samp(delta, 0.0)
        ci_lo, ci_hi = bootstrap_ci(delta, rng)
        result.update({
            "t": float(t), "df": int(len(delta) - 1), "p_ttest": float(p),
            "p_signflip": sign_flip_pvalue(delta, rng),
            "bootstrap_ci95_lo": ci_lo, "bootstrap_ci95_hi": ci_hi,
        })
    else:
        result.update({
            "t": np.nan, "df": np.nan, "p_ttest": np.nan, "p_signflip": np.nan,
            "bootstrap_ci95_lo": np.nan, "bootstrap_ci95_hi": np.nan,
        })
    subject_contrast["test"] = label
    subject_contrast["mapping_variant"] = mapping_variant
    return result, subject_contrast


def one_sample_test(values: np.ndarray, label: str, family: str, system: str,
                    rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    result = {
        "test": label, "family": family, "mapping_variant": "not_applicable",
        "left_condition": system, "right_condition": "zero",
        "n_subjects": int(len(values)),
        "mean_contrast": float(values.mean()) if len(values) else np.nan,
        "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else np.nan,
    }
    if len(values) >= 2:
        t, p = stats.ttest_1samp(values, 0.0)
        ci_lo, ci_hi = bootstrap_ci(values, rng)
        result.update({
            "t": float(t), "df": int(len(values) - 1), "p_ttest": float(p),
            "p_signflip": sign_flip_pvalue(values, rng),
            "bootstrap_ci95_lo": ci_lo, "bootstrap_ci95_hi": ci_hi,
        })
    else:
        result.update({"t": np.nan, "df": np.nan, "p_ttest": np.nan, "p_signflip": np.nan,
                       "bootstrap_ci95_lo": np.nan, "bootstrap_ci95_hi": np.nan})
    return result


def network_effects(tracts: pd.DataFrame, strict: bool) -> pd.DataFrame:
    eligible = tracts["is_network_strict"] if strict else tracts["is_network_labelled"]
    subset = tracts.loc[
        eligible & tracts["is_primary_system"] & tracts["is_cortical_pair"]
    ].copy()
    if subset.empty:
        return subset
    return subset.groupby(["subject", "tract_name", "pathway_type", "network_relation"], as_index=False).agg(
        aim2_interaction=("aim2_interaction", "mean"),
    )


def validate_inputs(tract_effects: pd.DataFrame, participant_system: pd.DataFrame) -> None:
    """Check that tract labels and participant-system averages agree."""
    validate_tract_classification(tract_effects)
    # Recalculate system averages to check consistency between input tables.
    expected_system = system_effects(tract_effects)
    keys = ["subject", "pathway_type"]
    expected_system = expected_system.set_index(keys).sort_index()
    saved_system = participant_system.set_index(keys).sort_index()
    if (
        not expected_system.index.equals(saved_system.index)
        or not np.allclose(expected_system["aim2_interaction"], saved_system["aim2_interaction"], atol=1e-12, rtol=0)
        or not np.array_equal(expected_system["n_tracts"], saved_system["n_tracts"])
    ):
        raise RuntimeError("Participant-system table disagrees with tract effects; rerun all of pt2_no12.")


def main() -> None:
    for path, label in [
        (TRACT_EFFECTS_PATH, "participant-tract effects"),
        (PARTICIPANT_SYSTEM_PATH, "participant-system effects"),
    ]:
        require_file(path, label)
    tract_effects = filter_subject_rows(pd.read_csv(TRACT_EFFECTS_PATH))
    participant_system = filter_subject_rows(pd.read_csv(PARTICIPANT_SYSTEM_PATH))
    required_tract = {
        "subject", "tract_name", "pathway_type", "is_primary_system", "is_cortical_pair",
        "is_network_labelled", "is_network_strict", "network_relation", "aim2_interaction",
        "n_units",
    }
    missing = required_tract - set(tract_effects.columns)
    if missing:
        raise RuntimeError(f"Tract-effect input lacks columns: {sorted(missing)}")
    validate_inputs(tract_effects, participant_system)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    results = []
    contrasts = []

    for system in ALL_SYSTEMS:
        values = participant_system.loc[
            participant_system["pathway_type"] == system, "aim2_interaction"
        ].to_numpy(dtype=float)
        results.append(one_sample_test(
            values, f"Tract-system effect vs 0: {system}", "tract_system_each", system, rng
        ))

    system_primary = participant_system.loc[
        participant_system["pathway_type"].isin(PLANNED_PAIR)
    ].copy()
    result, contrast = paired_test(
        system_primary, ["subject"], "pathway_type", "commissural", "association",
        "Tract-system contrast: commissural - association",
        "tract_system_primary", "not_applicable", rng,
    )
    results.append(result)
    contrasts.append(contrast)

    for strict, variant in [(False, "maximum_overlap_primary"), (True, "majority_overlap_sensitivity")]:
        network_tracts = network_effects(tract_effects, strict=strict)
        if network_tracts.empty:
            continue
        participant_network = network_tracts.groupby(["subject", "network_relation"], as_index=False).agg(
            aim2_interaction=("aim2_interaction", "mean")
        )
        result, contrast = paired_test(
            participant_network, ["subject"], "network_relation", "within", "between",
            "Canonical-network contrast: within - between",
            "network_secondary", variant, rng,
        )
        results.append(result)
        contrasts.append(contrast)

        participant_system_network = network_tracts.groupby(
            ["subject", "pathway_type", "network_relation"], as_index=False
        ).agg(aim2_interaction=("aim2_interaction", "mean"))
        within = participant_system_network.loc[
            participant_system_network["network_relation"] == "within"
        ].pivot_table(index=["subject", "pathway_type"], values="aim2_interaction", aggfunc="mean")
        between = participant_system_network.loc[
            participant_system_network["network_relation"] == "between"
        ].pivot_table(index=["subject", "pathway_type"], values="aim2_interaction", aggfunc="mean")
        system_delta = (within.rename(columns={"aim2_interaction": "within"}).join(
            between.rename(columns={"aim2_interaction": "between"}), how="inner"
        ))
        # An absent condition means no paired estimate, not a missing-column crash.
        system_delta = system_delta.reindex(columns=["within", "between"])
        system_delta["aim2_interaction"] = system_delta["within"] - system_delta["between"]
        result, contrast = paired_test(
            system_delta.reset_index(), ["subject"], "pathway_type", "commissural", "association",
            "Tract-system x canonical-network interaction",
            "network_secondary", variant, rng,
        )
        results.append(result)
        contrasts.append(contrast)

        for system in ALL_SYSTEMS:
            system_frame = participant_system_network.loc[
                participant_system_network["pathway_type"] == system
            ]
            result, contrast = paired_test(
                system_frame, ["subject"], "network_relation", "within", "between",
                f"Canonical-network contrast in {system}: within - between",
                "network_secondary", variant, rng,
            )
            results.append(result)
            contrasts.append(contrast)

    result_table = pd.DataFrame(results)
    result_table["p_fdr_bh"] = np.nan
    for _, index in result_table.groupby(["family", "mapping_variant"], dropna=False).groups.items():
        index = list(index)
        if result_table.loc[index, "family"].iloc[0] in {"network_secondary", "tract_system_each"}:
            result_table.loc[index, "p_fdr_bh"] = bh_adjust(result_table.loc[index, "p_ttest"])
    contrast_table = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    result_table.to_csv(OUTPUT_DIR / "aim2_tract_network_stats.csv", index=False)
    contrast_table.to_csv(OUTPUT_DIR / "aim2_tract_network_subject_contrasts.csv", index=False)
    with open(OUTPUT_DIR / "aim2_tract_network_stats.txt", "w", encoding="utf-8") as handle:
        handle.write(result_table.to_string(index=False))
        handle.write("\n\nInferential unit: participant-level paired contrasts.\n")
        handle.write("Primary mapping: maximum AAL3-to-Yeo7 SST overlap.\n")
        handle.write("Sensitivity mapping: majority Yeo overlap at both tract ends.\n")
    print(result_table.to_string(index=False))


if __name__ == "__main__":
    main()
