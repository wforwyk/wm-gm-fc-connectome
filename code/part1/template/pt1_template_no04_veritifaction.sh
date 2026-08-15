#!/bin/bash
# ============================================================
# pt1_template_no04_veritifaction.sh
# -----------------------------------------------------------
# PURPOSE : Verify atlas reslices and, when enabled, the TRACULA endpoint
#           PD reslices used by Part 2 Aim 2.
#
# CHECKS PER SUBJECT:
#   1. T1-, fMRI-, and DWI-space AAL files exist.
#   2. fMRI/DWI AAL files match their reference grids.
#   3. AAL label range is plausible.
#   4. Each produced fmri_endpt{1,2}.pd.nii matches the BnB analysis grid.
#
# NOTE : Endpoint images are continuous path-density maps. Their intensity
#        range is not checked here; only their fMRI-grid compatibility is.
# ============================================================

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS=("0001" "0002")
BASE_DIR="/path/to/your/project/derivatives/batch"
DARTEL_TEMPLATE_ID="YOUR_DARTEL_TEMPLATE"
SOURCE_SESSION="ses-01"
AAL_EXPECTED_MIN=100
AAL_EXPECTED_MAX=170

# Set true after running pt1_template_no05_coregister_endpoints_to_fmri.m.
VERIFY_ENDPOINTS=true
TRACT_DIR_SUFFIX="_avg16_syn_bbr"
# ────────────────────────────────────────────────────────────

WARN=()
get_shape() { fslinfo "$1" | awk '/^dim[123]/{printf $2" "}'; }
get_zooms() { fslinfo "$1" | awk '/^pixdim[123]/{printf "%.6f ", $2}'; }
get_max() { fslstats "$1" -R | awk '{printf "%d", $2}'; }

check_grid() {
    local image="$1" reference="$2" label="$3" subject="$4"
    local image_shape reference_shape image_zooms reference_zooms
    image_shape=$(get_shape "$image"); reference_shape=$(get_shape "$reference")
    image_zooms=$(get_zooms "$image"); reference_zooms=$(get_zooms "$reference")
    if [ "$image_shape" = "$reference_shape" ] && [ "$image_zooms" = "$reference_zooms" ]; then
        echo "  ${label} grid: OK"
    else
        echo "  ${label} grid: MISMATCH"
        echo "    image: shape=$image_shape zoom=$image_zooms"
        echo "    ref  : shape=$reference_shape zoom=$reference_zooms"
        WARN+=("[$subject] ${label} grid mismatch: $(basename "$image")")
    fi
}

echo "============================================"
echo "  Template and endpoint-reslice verification"
echo "  Subjects: ${#SUBJECTS[@]}"
echo "============================================"

for subj in "${SUBJECTS[@]}"; do
    der_dir="${BASE_DIR}/${subj}/gm/derived"
    func_dir="${BASE_DIR}/${subj}/gm/func/preprocessing"
    dwi_dir="${BASE_DIR}/${subj}/wm/freesurfer/${subj}/dmri"
    dpath_dir="${BASE_DIR}/${subj}/wm/freesurfer/${subj}/dpath"
    suffix="u_rc1sub-${subj}_${SOURCE_SESSION}_T1w_${DARTEL_TEMPLATE_ID}.nii"
    f_t1="${der_dir}/wsst_all_ROIs_AAL_${suffix}"
    f_fmri="${der_dir}/fmri_wsst_all_ROIs_AAL_${suffix}"
    f_dwi="${der_dir}/dwi_wsst_all_ROIs_AAL_${suffix}"
    f_fmri_ref="${func_dir}/mean${subj}_rest.nii"
    f_bnb_ref="${BASE_DIR}/${subj}/gm/func/derived/bnbmdenoised_detrended_rest_${subj}.nii"
    f_dwi_ref="${dwi_dir}/dwi.nii"

    echo; echo "[$subj]"
    for f in "$f_t1" "$f_fmri" "$f_dwi"; do
        if [ ! -f "$f" ]; then
            echo "  MISSING: $(basename "$f")"
            WARN+=("[$subj] missing AAL file: $(basename "$f")")
        fi
    done

    if [ -f "$f_fmri" ] && [ -f "$f_fmri_ref" ]; then
        check_grid "$f_fmri" "$f_fmri_ref" "fMRI AAL" "$subj"
    fi
    if [ -f "$f_dwi" ] && [ -f "$f_dwi_ref" ]; then
        check_grid "$f_dwi" "$f_dwi_ref" "DWI AAL" "$subj"
    fi
    for f in "$f_t1" "$f_fmri" "$f_dwi"; do
        [ -f "$f" ] || continue
        maxval=$(get_max "$f")
        if [ "$maxval" -lt "$AAL_EXPECTED_MIN" ] || [ "$maxval" -gt "$AAL_EXPECTED_MAX" ]; then
            echo "  label max: WARN ($maxval) — $(basename "$f")"
            WARN+=("[$subj] AAL label max out of range: $(basename "$f")")
        fi
    done

    if [ "$VERIFY_ENDPOINTS" = true ]; then
        if [ ! -d "$dpath_dir" ] || [ ! -f "$f_bnb_ref" ]; then
            echo "  endpoint check: cannot run (missing dpath directory or BnB reference)"
            WARN+=("[$subj] endpoint verification unavailable")
            continue
        fi
        shopt -s nullglob
        tract_dirs=("${dpath_dir}"/*"${TRACT_DIR_SUFFIX}")
        shopt -u nullglob
        for tract_dir in "${tract_dirs[@]}"; do
            [ -d "$tract_dir" ] || continue
            for endpoint in 1 2; do
                endpoint_file="${tract_dir}/fmri_endpt${endpoint}.pd.nii"
                if [ -f "$endpoint_file" ]; then
                    check_grid "$endpoint_file" "$f_bnb_ref" "endpoint $(basename "$tract_dir") endpt${endpoint}" "$subj"
                else
                    echo "  MISSING endpoint: $(basename "$tract_dir")/fmri_endpt${endpoint}.pd.nii"
                    WARN+=("[$subj] missing fMRI endpoint: $(basename "$tract_dir") endpt${endpoint}")
                fi
            done
        done
    fi
done

echo; echo "============================================"
if [ ${#WARN[@]} -eq 0 ]; then
    echo "  All requested checks passed."
else
    echo "  WARNINGS (${#WARN[@]}):"
    printf '    %s\n' "${WARN[@]}"
fi
echo "============================================"
