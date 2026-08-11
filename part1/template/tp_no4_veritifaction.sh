#!/bin/bash
# ============================================================
# no4_verification.sh
# -----------------------------------------------------------
# PURPOSE : Verify that fmri_ and dwi_ coregistered AAL files
#           have correct shape, voxel size, and label range.
#
# CHECKS PER SUBJECT:
#   1. All 3 files exist (wsst_, fmri_, dwi_)
#   2. fmri_ shape/voxel matches mean{subj}_rest.nii (fMRI ref)
#   3. dwi_  shape/voxel matches dwi.nii (DWI ref)
#   4. AAL label max in valid range (100~170)
#      AAL2=116, AAL3=166 — both accepted
#
# OUTPUT  : Summary printed to terminal + warning list at end
#
# NOTE    : Requires FSL (fslinfo, fslstats).
#           Edit SUBJECTS in USER SETTINGS below.
# ============================================================

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS=("1001" "1002") # e.g. subject id must be in " " and divided by space

BASE_DIR="/your/designated/base/directory"

AAL_EXPECTED_MIN=100  
AAL_EXPECTED_MAX=170
# ────────────────────────────────────────────────────────────

WARN=()

get_shape() { fslinfo "$1" | awk '/^dim[123]/{printf $2" "}'; }
get_zooms() { fslinfo "$1" | awk '/^pixdim[123]/{printf "%.2f", $2}'; }
get_max()   { fslstats "$1" -R | awk '{printf "%d", $2}'; }

echo "============================================"
echo "  no4_verification.sh"
echo "  Subjects : ${#SUBJECTS[@]}"
echo "  AAL label range: ${AAL_EXPECTED_MIN}~${AAL_EXPECTED_MAX} (AAL2=116, AAL3=166)"
echo "============================================"

for subj in "${SUBJECTS[@]}"; do
    der_dir="${BASE_DIR}/${subj}/gm/derived"
    func_dir="${BASE_DIR}/${subj}/gm/func/preprocessing"
    dwi_dir="${BASE_DIR}/${subj}/wm/freesurfer/${subj}/dmri"

    suffix="sub-${subj}_ses-1_T1w_NCKU_336Ss.nii"
    f_t1="${der_dir}/wsst_all_ROIs_AAL_u_rc1${suffix}"
    f_fmri="${der_dir}/fmri_wsst_all_ROIs_AAL_u_rc1${suffix}"
    f_dwi="${der_dir}/dwi_wsst_all_ROIs_AAL_u_rc1${suffix}"
    f_fmri_ref="${func_dir}/mean${subj}_rest.nii"
    f_dwi_ref="${dwi_dir}/dwi.nii"

    echo ""
    echo "[$subj]"

    # ── 1. verify if the file exists ─────────────────────────────────
    ok=1
    for f in "$f_t1" "$f_fmri" "$f_dwi"; do
        if [ ! -f "$f" ]; then
            echo "  MISSING : $(BASE_DIRname $f)"
            WARN+=("[$subj] MISSING: $(BASE_DIRname $f)")
            ok=0
        fi
    done
    [ $ok -eq 0 ] && continue

    # ── 2. fmri_ vs fMRI reference ────────────────────────
    if [ -f "$f_fmri_ref" ]; then
        shape_fmri=$(get_shape "$f_fmri")
        shape_ref=$(get_shape "$f_fmri_ref")
        zoom_fmri=$(get_zooms "$f_fmri")
        zoom_ref=$(get_zooms "$f_fmri_ref")

        if [ "$shape_fmri" = "$shape_ref" ] && [ "$zoom_fmri" = "$zoom_ref" ]; then
            echo "  fmri_ shape/zoom : OK  [$shape_fmri/ ${zoom_fmri}]"
        else
            echo "  fmri_ shape/zoom : MISMATCH"
            echo "    fmri_  : shape=$shape_fmri zoom=$zoom_fmri"
            echo "    ref    : shape=$shape_ref  zoom=$zoom_ref"
            WARN+=("[$subj] fmri_ shape/zoom mismatch")
        fi
    else
        echo "  fmri_ref not found, skipping shape check"
        WARN+=("[$subj] fMRI reference not found: $f_fmri_ref")
    fi

    # ── 3. dwi_ vs DWI reference ──────────────────────────
    if [ -f "$f_dwi_ref" ]; then
        shape_dwi=$(get_shape "$f_dwi")
        shape_dref=$(get_shape "$f_dwi_ref")
        zoom_dwi=$(get_zooms "$f_dwi")
        zoom_dref=$(get_zooms "$f_dwi_ref")

        if [ "$shape_dwi" = "$shape_dref" ] && [ "$zoom_dwi" = "$zoom_dref" ]; then
            echo "  dwi_  shape/zoom : OK  [$shape_dwi/ ${zoom_dwi}]"
        else
            echo "  dwi_  shape/zoom : MISMATCH"
            echo "    dwi_   : shape=$shape_dwi  zoom=$zoom_dwi"
            echo "    ref    : shape=$shape_dref zoom=$zoom_dref"
            WARN+=("[$subj] dwi_ shape/zoom mismatch")
        fi
    else
        echo "  dwi_ref not found, skipping shape check"
        WARN+=("[$subj] DWI reference not found: $f_dwi_ref")
    fi

    # ── 4. AAL label max (AAL2=116, AAL3=166) ────────────
    for f in "$f_t1" "$f_fmri" "$f_dwi"; do
        maxval=$(get_max "$f")
        fname=$(BASE_DIRname "$f")
        if [ "$maxval" -ge "$AAL_EXPECTED_MIN" ] && [ "$maxval" -le "$AAL_EXPECTED_MAX" ]; then
            echo "  label max : OK  ($maxval) — $fname"
        else
            echo "  label max : WARN ($maxval, expected ${AAL_EXPECTED_MIN}~${AAL_EXPECTED_MAX}) — $fname"
            WARN+=("[$subj] label max=$maxval out of expected range: $fname")
        fi
    done

done

# ── summary ─────────────────────────────────────────────
echo ""
echo "============================================"
if [ ${#WARN[@]} -eq 0 ]; then
    echo "  All subjects passed. No issues found."
else
    echo "  WARNINGS (${#WARN[@]}) — check again manually:"
    for w in "${WARN[@]}"; do
        echo "    $w"
    done
fi
echo "============================================"