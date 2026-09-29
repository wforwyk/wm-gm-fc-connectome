#!/usr/bin/env python3
"""
pt2_no11_yeo7_aal_lookup.py
============================
Create a reproducible AAL3-to-Yeo7 lookup in the project SST space.

This is a reference-table stage, not a participant-level FC analysis.  It
resamples the already SST-warped, categorical Yeo 7-network map onto the
existing SST AAL3 grid with nearest-neighbour interpolation, then records the
complete overlap profile of every AAL3 ROI.  Aim 2 follow-up scripts join the
resulting lookup by ROI ID; they do not re-register a Yeo map for every
participant.

Inputs
------
1. sst_all_ROIs_AAL.nii: current project AAL3 atlas in SST space.
2. sst_Yeo2011_...LiberalMask.nii: output of template no07.
3. aal3_region_category.csv: cortical/non-cortical status of each AAL3 ROI.

Outputs
-------
- yeo7_liberal_on_aal3_sst_grid.nii.gz
- aal3_yeo7_overlap_lookup.csv
- aal3_yeo7_overlap_long.csv
- yeo7_aal3_sst_qc.png
- yeo7_aal3_lookup_manifest.json

Coverage and dominance columns support maximum-overlap and
majority-overlap mapping in no12/no13.
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy import ndimage


# =============================================================================
# USER SETTINGS
# =============================================================================
SST_TEMPLATE_DIR = Path(
    "/path/to/your/project/"
    "derivatives/template/sst_atlas"
)
YEO_TEMPLATE_DIR = Path(
    "/path/to/your/project/"
    "derivatives/template/Yeo_JNeurophysiol11_MNI152"
)
AAL3_TEMPLATE_DIR = Path(
    "/path/to/your/project/"
    "derivatives/template/AAL3"
)
SST_AAL_PATH = SST_TEMPLATE_DIR / "sst_all_ROIs_AAL.nii"
SST_YEO_PATH = (
    YEO_TEMPLATE_DIR / "sst" /
    "sst_Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii"
)
REGION_CATEGORY_PATH = AAL3_TEMPLATE_DIR / "aal3_region_category.csv"
OUTPUT_DIR = Path(
    "/path/to/your/project/"
    "derivatives/results/part2/no11_yeo7_aal_lookup"
)
# =============================================================================


YEO7_NAMES = {
    1: "Visual",
    2: "Somatomotor",
    3: "Dorsal_Attention",
    4: "Ventral_Attention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}
YEO7_COLORS = [
    "#000000", "#781286", "#4682B4", "#00760E", "#C43AFA",
    "#DCF8A4", "#E69422", "#CD3E4E",
]


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def categorical_labels(image: nib.spatialimages.SpatialImage, label: str) -> np.ndarray:
    """Load a discrete atlas without inventing labels by interpolation."""
    data = np.asarray(image.get_fdata(), dtype=float)
    if data.ndim != 3:
        raise ValueError(f"{label} must be a 3-D label map; got shape {data.shape}.")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError(f"{label} contains no finite values.")
    if not np.allclose(finite, np.rint(finite), atol=1e-4):
        raise ValueError(f"{label} is not categorical; do not use a trilinearly resampled atlas.")
    return np.rint(data).astype(np.int16)


def resample_yeo_to_aal(yeo_img, aal_img) -> np.ndarray:
    """Nearest-neighbour resampling onto the exact SST AAL3 voxel grid."""
    same_grid = (
        yeo_img.shape == aal_img.shape
        and np.allclose(yeo_img.affine, aal_img.affine, atol=1e-5)
    )
    if same_grid:
        return categorical_labels(yeo_img, "SST Yeo map")
    resampled = resample_from_to(yeo_img, (aal_img.shape, aal_img.affine), order=0)
    return categorical_labels(resampled, "nearest-neighbour Yeo map on SST AAL3 grid")


def save_qc_mosaic(aal: np.ndarray, yeo: np.ndarray, output_path: Path) -> None:
    """Save a qualitative SST overlay: Yeo labels with AAL borders."""
    nonzero_z = np.flatnonzero(np.any(yeo > 0, axis=(0, 1)))
    if nonzero_z.size == 0:
        raise ValueError("Yeo SST map has no labelled cortical voxels after resampling.")
    positions = np.linspace(0, nonzero_z.size - 1, 6).round().astype(int)
    slices = nonzero_z[positions]
    boundaries = (aal > 0) & ~ndimage.binary_erosion(aal > 0)
    cmap = ListedColormap(YEO7_COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for axis, z in zip(axes.ravel(), slices):
        axis.imshow(yeo[:, :, z].T, origin="lower", cmap=cmap, vmin=0, vmax=7)
        border = np.ma.masked_where(~boundaries[:, :, z].T, boundaries[:, :, z].T)
        axis.imshow(border, origin="lower", cmap=ListedColormap(["white"]), alpha=0.72)
        axis.set_title(f"SST axial slice z={z}")
        axis.set_axis_off()
    fig.suptitle("Yeo 7-network labels on the SST AAL3 grid; white = AAL3 boundaries", fontsize=12)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for path, label in [
        (SST_AAL_PATH, "SST AAL3 map"),
        (SST_YEO_PATH, "SST Yeo 7-network map"),
        (REGION_CATEGORY_PATH, "AAL3 category table"),
    ]:
        require_file(path, label)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aal_img = nib.load(str(SST_AAL_PATH))
    yeo_img = nib.load(str(SST_YEO_PATH))
    aal = categorical_labels(aal_img, "SST AAL3 map")
    yeo = resample_yeo_to_aal(yeo_img, aal_img)
    allowed = set(range(8))
    observed = set(np.unique(yeo).astype(int).tolist())
    unexpected = observed - allowed
    if unexpected:
        raise ValueError(f"Yeo map contains unexpected labels {sorted(unexpected)}; expected only 0..7.")
    missing_networks = set(YEO7_NAMES) - observed
    if missing_networks:
        raise ValueError(f"Yeo SST map is missing network labels {sorted(missing_networks)}.")

    resampled_path = OUTPUT_DIR / "yeo7_liberal_on_aal3_sst_grid.nii.gz"
    header = aal_img.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(yeo.astype(np.int16), aal_img.affine, header), str(resampled_path))

    categories = pd.read_csv(REGION_CATEGORY_PATH)
    required_category_columns = {"roi_id", "roi_name", "category", "include"}
    missing = required_category_columns - set(categories.columns)
    if missing:
        raise ValueError(f"AAL3 category table is missing columns: {sorted(missing)}")

    records = []
    long_records = []
    for roi in categories.itertuples(index=False):
        roi_id = int(roi.roi_id)
        roi_mask = aal == roi_id
        n_aal = int(roi_mask.sum())
        labels = yeo[roi_mask]
        labelled = labels[labels > 0]
        counts = {network_id: int((labelled == network_id).sum()) for network_id in YEO7_NAMES}
        n_labelled = int(labelled.size)
        if n_labelled:
            dominant_id = max(counts, key=counts.get)
            dominant_n = counts[dominant_id]
            dominant_share = dominant_n / n_labelled
        else:
            dominant_id = np.nan
            dominant_n = 0
            dominant_share = np.nan
        record = {
            "roi_id": roi_id,
            "roi_name": roi.roi_name,
            "category": roi.category,
            "include": int(roi.include),
            "is_cortical": bool(roi.category == "cortical"),
            "n_aal3_voxels": n_aal,
            "n_yeo_labelled_voxels": n_labelled,
            "yeo_coverage_of_aal3": n_labelled / n_aal if n_aal else np.nan,
            "dominant_yeo_id": dominant_id,
            "dominant_yeo_name": YEO7_NAMES.get(dominant_id, np.nan),
            "dominant_yeo_voxels": dominant_n,
            "dominant_yeo_share_of_labelled": dominant_share,
            "is_majority_mapped": bool(n_labelled and dominant_share >= 0.50),
        }
        records.append(record)
        for network_id, count in counts.items():
            long_records.append({
                "roi_id": roi_id,
                "roi_name": roi.roi_name,
                "category": roi.category,
                "yeo_id": network_id,
                "yeo_name": YEO7_NAMES[network_id],
                "n_overlap_voxels": count,
                "share_of_yeo_labelled_voxels": count / n_labelled if n_labelled else np.nan,
                "share_of_aal3_voxels": count / n_aal if n_aal else np.nan,
            })

    lookup = pd.DataFrame(records).sort_values("roi_id")
    overlap_long = pd.DataFrame(long_records).sort_values(["roi_id", "yeo_id"])
    lookup.to_csv(OUTPUT_DIR / "aal3_yeo7_overlap_lookup.csv", index=False)
    overlap_long.to_csv(OUTPUT_DIR / "aal3_yeo7_overlap_long.csv", index=False)
    save_qc_mosaic(aal, yeo, OUTPUT_DIR / "yeo7_aal3_sst_qc.png")

    cortical = lookup.query("is_cortical")
    manifest = {
        "sst_aal_path": str(SST_AAL_PATH),
        "sst_yeo_path": str(SST_YEO_PATH),
        "resampled_yeo_path": str(resampled_path),
        "interpolation": "nearest-neighbour (order=0)",
        "yeo_labels": YEO7_NAMES,
        "n_aal3_rois": int(len(lookup)),
        "n_cortical_aal3_rois": int(len(cortical)),
        "n_cortical_rois_with_yeo_label": int(cortical["dominant_yeo_id"].notna().sum()),
        "n_cortical_rois_with_majority_label": int(cortical["is_majority_mapped"].sum()),
    }
    with open(OUTPUT_DIR / "yeo7_aal3_lookup_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
