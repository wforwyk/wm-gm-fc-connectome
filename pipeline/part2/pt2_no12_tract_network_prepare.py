#!/usr/bin/env python3
"""
pt2_no12_tract_network_prepare.py
==================================
Prepare participant-level tract-system and Yeo-network effects.

Read no09 matched-FC rows and retain endpoint units marked eligible_primary
in no10. Reconstruct endpoint x distant-versus-adjacent interactions and check
participant averages against no10 before adding tract and network labels.

System effects include commissural, projection, association and other
(bilateral fornix) tracts. Average eligible endpoint units within each tract,
then average tracts equally within each participant and system.

Network effects require both regions to be cortical and labelled by Yeo7.
Projection tracts contribute only through their cortical-cortical units.

"""
import os
from pathlib import Path

import numpy as np
import pandas as pd


from cohort_settings import SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, select_subjects, select_files, filter_subject_rows

# =============================================================================
# USER SETTINGS
# =============================================================================
NO09_INPUT_DIR = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no09_matched_endpoint_fc"
)
NO10_BALANCE_PATH = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no10_matched_endpoint_stats/"
    "aim2_matching_balance_all.csv"
)
NO10_SUBJECT_INTERACTION_PATH = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no10_matched_endpoint_stats/"
    "aim2_subject_interaction.csv"
)
LOOKUP_PATH = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no11_yeo7_aal_lookup/"
    "aal3_yeo7_overlap_lookup.csv"
)
OUTPUT_DIR = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no12_tract_network_prepare"
)
# A majority mapping is not required for the main maximum-overlap lookup.
# It defines the strict mapping sensitivity subset.
STRICT_DOMINANT_SHARE = 0.50
RECONCILIATION_TOLERANCE = 1e-10
# =============================================================================


# Tract classification: bilateral fornix is 'other'.
# TRACULA reference: Maffei et al. (2021), NeuroImage 245:118706.
# MCP is classified as other but excluded upstream from group analysis.
TRACT_SYSTEM = {
    "commissural": {
        "acomm", "cc.genu", "cc.rostrum",
        "cc.splenium", "cc.bodyc", "cc.bodyp", "cc.bodypf", "cc.bodypm",
        "cc.bodyt",
    },
    "other": {"mcp", "lh.fx", "rh.fx"},
    "projection": {"lh.atr", "rh.atr", "lh.ar", "rh.ar", "lh.or", "rh.or"},
    "association": {
        "lh.af", "rh.af", "lh.slf1", "rh.slf1", "lh.slf2", "rh.slf2",
        "lh.slf3", "rh.slf3", "lh.uf", "rh.uf", "lh.ilf", "rh.ilf",
        "lh.mlf", "rh.mlf", "lh.emc", "rh.emc", "lh.fat", "rh.fat",
        "lh.cbd", "rh.cbd", "lh.cbv", "rh.cbv",
    },
}
PRIMARY_SYSTEMS = {"commissural", "association", "projection", "other"}


def system_effects(tract_effects: pd.DataFrame) -> pd.DataFrame:
    """Participant-system effects over all no10-eligible units (no cortical filter).

    tract_effects is keyed by is_cortical_pair, so one participant-tract can occupy
    two rows.  Rows are recombined with unit weights so each participant-tract
    effect equals the mean over all of its eligible endpoint units, then tracts are
    averaged with equal weight within participant and system.
    """
    rows = tract_effects.loc[tract_effects["is_primary_system"].astype(bool)].copy()
    rows["_weighted"] = rows["aim2_interaction"] * rows["n_units"]
    per_tract = rows.groupby(["subject", "tract_name", "pathway_type"], as_index=False).agg(
        _weighted=("_weighted", "sum"), n_units=("n_units", "sum")
    )
    per_tract["aim2_interaction"] = per_tract["_weighted"] / per_tract["n_units"]
    return per_tract.groupby(["subject", "pathway_type"], as_index=False).agg(
        aim2_interaction=("aim2_interaction", "mean"),
        n_tracts=("tract_name", "nunique"),
    )


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_nonempty_csvs(pattern: str) -> pd.DataFrame:
    frames = []
    for file in select_files(sorted(NO09_INPUT_DIR.glob(pattern))):
        try:
            frame = pd.read_csv(file)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No non-empty files matched {NO09_INPUT_DIR / pattern}")
    return pd.concat(frames, ignore_index=True)


def pathway_type(tract: str) -> str:
    for system, tracts in TRACT_SYSTEM.items():
        if tract in tracts:
            return system
    return "other"


def require_unique_rows(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, keys].head(5).to_dict("records")
        raise RuntimeError(f"{label} has duplicate rows for a unit expected to be unique: {example}")


def validate_tract_classification(frame: pd.DataFrame) -> None:
    """Validate tract labels and system membership."""
    expected = frame["tract_name"].map(pathway_type)
    invalid = (
        frame["pathway_type"].ne(expected)
        | frame["is_primary_system"].ne(expected.isin(PRIMARY_SYSTEMS))
    )
    if invalid.any():
        names = sorted(frame.loc[invalid, "tract_name"].unique())
        raise RuntimeError(
            f"Stale/inconsistent tract classification for {names}. "
            "Regenerate no12 tables before running no13."
        )


def attach_lookup(units: pd.DataFrame, lookup: pd.DataFrame, roi_column: str, prefix: str) -> pd.DataFrame:
    columns = [
        "roi_id", "roi_name", "category", "is_cortical", "n_yeo_labelled_voxels",
        "yeo_coverage_of_aal3", "dominant_yeo_id", "dominant_yeo_name",
        "dominant_yeo_share_of_labelled",
    ]
    right = lookup.loc[:, columns].rename(columns={
        "roi_id": roi_column,
        "roi_name": f"{prefix}_roi_name",
        "category": f"{prefix}_category",
        "is_cortical": f"{prefix}_is_cortical",
        "n_yeo_labelled_voxels": f"{prefix}_n_yeo_labelled_voxels",
        "yeo_coverage_of_aal3": f"{prefix}_yeo_coverage",
        "dominant_yeo_id": f"{prefix}_yeo_id",
        "dominant_yeo_name": f"{prefix}_yeo_name",
        "dominant_yeo_share_of_labelled": f"{prefix}_yeo_dominant_share",
    })
    out = units.merge(right, on=roi_column, how="left", validate="many_to_one")
    if out[f"{prefix}_category"].isna().any():
        missing = sorted(out.loc[out[f"{prefix}_category"].isna(), roi_column].dropna().unique())
        raise RuntimeError(f"AAL3 ROI IDs are absent from the Yeo lookup: {missing}")
    return out


def make_unit_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Recreate no10's interaction before it discarded target_roi."""
    unit_keys = ["subject", "tract_name", "endpoint", "source_roi", "target_roi"]
    class_key = [*unit_keys, "voxel_class"]
    remote = data.loc[data["target_type"] == "wm_connected_distant", class_key + ["mean_fc_z"]].copy()
    require_unique_rows(remote, class_key, "wm_connected_distant rows")
    remote_wide = (
        remote.pivot(index=unit_keys, columns="voxel_class", values="mean_fc_z")
        .rename(columns={
            "endpoint": "endpoint_distant",
            "matched_non_endpoint": "matched_non_endpoint_distant",
        })
        .reset_index()
    )

    adjacent_keys = ["subject", "tract_name", "endpoint", "source_roi", "voxel_class"]
    adjacent = (
        data.loc[data["target_type"] == "physically_adjacent", adjacent_keys + ["mean_fc_z"]]
        .groupby(adjacent_keys, as_index=False)
        .agg(adjacent_mean_fc_z=("mean_fc_z", "mean"))
    )
    adjacent_wide = (
        adjacent.pivot(
            index=["subject", "tract_name", "endpoint", "source_roi"],
            columns="voxel_class", values="adjacent_mean_fc_z"
        )
        .rename(columns={
            "endpoint": "endpoint_adjacent",
            "matched_non_endpoint": "matched_non_endpoint_adjacent",
        })
        .reset_index()
    )
    units = remote_wide.merge(
        adjacent_wide,
        on=["subject", "tract_name", "endpoint", "source_roi"],
        how="inner", validate="one_to_one",
    )
    needed = {
        "endpoint_distant", "matched_non_endpoint_distant",
        "endpoint_adjacent", "matched_non_endpoint_adjacent",
    }
    if not needed.issubset(units.columns):
        raise RuntimeError(f"Could not form endpoint/control interaction columns: {sorted(needed - set(units.columns))}")
    units["endpoint_distant_minus_adjacent"] = units["endpoint_distant"] - units["endpoint_adjacent"]
    units["control_distant_minus_adjacent"] = (
        units["matched_non_endpoint_distant"] - units["matched_non_endpoint_adjacent"]
    )
    units["aim2_interaction"] = (
        units["endpoint_distant_minus_adjacent"] - units["control_distant_minus_adjacent"]
    )
    return units


def reconcile_no10(units: pd.DataFrame) -> pd.DataFrame:
    no10 = filter_subject_rows(pd.read_csv(NO10_SUBJECT_INTERACTION_PATH))
    needed = {"subject", "endpoint_specific_distant_preference"}
    if not needed.issubset(no10.columns):
        raise RuntimeError(f"no10 subject interaction table lacks columns {sorted(needed - set(no10.columns))}")
    ours = units.groupby("subject", as_index=False).agg(
        reconstructed_interaction=("aim2_interaction", "mean")
    )
    check = ours.merge(
        no10.loc[:, ["subject", "endpoint_specific_distant_preference"]],
        on="subject", how="outer", validate="one_to_one",
    )
    check["difference_from_no10"] = (
        check["reconstructed_interaction"] - check["endpoint_specific_distant_preference"]
    )
    valid = check.dropna(subset=["difference_from_no10"])
    if len(valid) != len(check) or not np.allclose(
        valid["difference_from_no10"], 0.0, atol=RECONCILIATION_TOLERANCE, rtol=0.0
    ):
        raise RuntimeError(
            "Reconstructed interactions do not reproduce the no10 subject effects. "
            "Inspect target aggregation before interpreting any tract/network result."
        )
    return check


def summarise_support(units: pd.DataFrame, tract_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in [
        ("all_no10_eligible_units", units),
        ("system_analysis_all_classes", units.query("is_primary_system")),
        ("cortico_cortical_all_classes", units.query("is_cortical_pair and is_primary_system")),
        ("yeo_max_overlap_primary", units.query("is_network_labelled and is_cortical_pair and is_primary_system")),
        ("yeo_majority_sensitivity", units.query("is_network_strict and is_cortical_pair and is_primary_system")),
    ]:
        rows.append({
            "stage": label,
            "n_units": int(len(frame)),
            "n_subjects": int(frame["subject"].nunique()),
            "n_tracts": int(frame["tract_name"].nunique()),
        })
    for (system, relation), frame in tract_effects.loc[
        tract_effects["is_network_labelled"] & tract_effects["is_primary_system"]
    ].groupby(["pathway_type", "network_relation"], dropna=False):
        rows.append({
            "stage": "yeo_max_overlap_tract_support",
            "pathway_type": system,
            "network_relation": relation,
            "n_units": int(len(frame)),
            "n_subjects": int(frame["subject"].nunique()),
            "n_tracts": int(frame["tract_name"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for path, label in [
        (NO10_BALANCE_PATH, "no10 matching eligibility table"),
        (NO10_SUBJECT_INTERACTION_PATH, "no10 subject interaction table"),
        (LOOKUP_PATH, "AAL3-to-Yeo7 lookup"),
    ]:
        require_file(path, label)
    if not NO09_INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Missing no09 input directory: {NO09_INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = read_nonempty_csvs("*_aim2_matched_endpoint_fc.csv").dropna(subset=["mean_fc_z"])
    required = {
        "subject", "tract_name", "endpoint", "source_roi", "target_roi",
        "target_type", "voxel_class", "mean_fc_z",
    }
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"no09 matched-FC files lack columns: {sorted(missing)}")
    balance = filter_subject_rows(pd.read_csv(NO10_BALANCE_PATH))
    needed_balance = {"subject", "tract", "endpoint", "eligible_primary"}
    missing_balance = needed_balance - set(balance.columns)
    if missing_balance:
        raise RuntimeError(f"no10 eligibility table lacks columns: {sorted(missing_balance)}")
    eligible = balance.loc[balance["eligible_primary"].astype(bool), ["subject", "tract", "endpoint"]]
    eligible = eligible.rename(columns={"tract": "tract_name"})
    data = data.merge(eligible, on=["subject", "tract_name", "endpoint"], how="inner", validate="many_to_one")
    if data.empty:
        raise RuntimeError("No no10-eligible matched FC rows remain.")

    units = make_unit_interactions(data)
    reconciliation = reconcile_no10(units)
    lookup = pd.read_csv(LOOKUP_PATH)
    required_lookup = {
        "roi_id", "roi_name", "category", "is_cortical", "n_yeo_labelled_voxels",
        "yeo_coverage_of_aal3", "dominant_yeo_id", "dominant_yeo_name",
        "dominant_yeo_share_of_labelled",
    }
    missing_lookup = required_lookup - set(lookup.columns)
    if missing_lookup:
        raise RuntimeError(f"Yeo lookup lacks columns: {sorted(missing_lookup)}")
    units = attach_lookup(units, lookup, "source_roi", "source")
    units = attach_lookup(units, lookup, "target_roi", "target")
    units["pathway_type"] = units["tract_name"].map(pathway_type)
    units["is_primary_system"] = units["pathway_type"].isin(PRIMARY_SYSTEMS)
    # ROI-category check only; this does not establish tract anatomy/alignment.
    units["is_cortical_pair"] = units["source_is_cortical"] & units["target_is_cortical"]
    validate_tract_classification(units)
    units["is_network_labelled"] = (
        units["is_cortical_pair"]
        & units["source_yeo_id"].notna()
        & units["target_yeo_id"].notna()
    )
    units["network_relation"] = np.where(
        units["is_network_labelled"],
        np.where(units["source_yeo_id"] == units["target_yeo_id"], "within", "between"),
        pd.NA,
    )
    units["is_network_strict"] = (
        units["is_network_labelled"]
        & units["source_yeo_dominant_share"].ge(STRICT_DOMINANT_SHARE)
        & units["target_yeo_dominant_share"].ge(STRICT_DOMINANT_SHARE)
    )

    tract_keys = ["subject", "tract_name", "pathway_type", "is_primary_system", "is_cortical_pair"]
    tract_effects = units.groupby(tract_keys, as_index=False).agg(
        aim2_interaction=("aim2_interaction", "mean"),
        n_units=("aim2_interaction", "size"),
        n_tract_ends=("endpoint", "nunique"),
        is_network_labelled=("is_network_labelled", "all"),
        is_network_strict=("is_network_strict", "all"),
        network_relation=("network_relation", lambda x: x.dropna().iloc[0] if len(x.dropna()) else pd.NA),
        n_network_relations=("network_relation", lambda x: x.dropna().nunique()),
    )
    inconsistent = tract_effects["n_network_relations"].gt(1)
    if inconsistent.any():
        example = tract_effects.loc[inconsistent, tract_keys + ["n_network_relations"]].head(5).to_dict("records")
        raise RuntimeError(f"A tract has inconsistent Yeo within/between labels across ends: {example}")

    participant_system = system_effects(tract_effects)
    primary_tracts = tract_effects.query("is_cortical_pair and is_primary_system")
    network_tracts = primary_tracts.loc[primary_tracts["is_network_labelled"]].copy()
    participant_system_network = network_tracts.groupby(
        ["subject", "pathway_type", "network_relation"], as_index=False
    ).agg(aim2_interaction=("aim2_interaction", "mean"), n_tracts=("tract_name", "nunique"))

    support = summarise_support(units, tract_effects)
    units.to_csv(OUTPUT_DIR / "aim2_tract_network_units_all.csv", index=False)
    tract_effects.to_csv(OUTPUT_DIR / "aim2_tract_effects.csv", index=False)
    participant_system.to_csv(OUTPUT_DIR / "aim2_participant_system_effects.csv", index=False)
    participant_system_network.to_csv(OUTPUT_DIR / "aim2_participant_system_network_effects.csv", index=False)
    reconciliation.to_csv(OUTPUT_DIR / "aim2_no10_reconciliation.csv", index=False)
    support.to_csv(OUTPUT_DIR / "aim2_tract_network_support.csv", index=False)
    print(
        "Prepared tract/network effects: "
        f"{len(units)} endpoint units -> {len(tract_effects)} participant-tract effects; "
        f"{participant_system.subject.nunique()} participants with tract-system support (all four classes)."
    )


if __name__ == "__main__":
    main()
