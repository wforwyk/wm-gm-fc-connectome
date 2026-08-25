#!/usr/bin/env bash
# =============================================================================
# pt1_template_no06_verify_dwi_fmri_alignment.sh
#
# PURPOSE:
# Quantify the DWI-to-fMRI alignment assumption used when TRACULA endpoint PD
# maps are resliced onto the BnB functional grid. For each subject, estimate a
# 6-DOF normalized-mutual-information FLIRT registration from the DWI reference
# volume to the mean fMRI, then express it relative to the mapping implied by
# the NIfTI qform/sform headers.
#
# This is QC only: it never changes analysis images or registration transforms.
# =============================================================================

set -euo pipefail

# =============================================================================
# USER SETTINGS
# =============================================================================
# Leave empty to process every numeric subject directory under BASE_DIR.
SUBJECTS=()
BASE_DIR="/path/to/your/project/derivatives/batch"
DWI_TEMPLATE="{base}/{subject}/wm/freesurfer/{subject}/dmri/dwi.nii"
FMRI_TEMPLATE="{base}/{subject}/gm/func/preprocessing/mean{subject}_rest.nii"
OUTPUT_SUBDIR="gm/anat/derived/dwi_fmri_alignment_qc"
# =============================================================================

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'ERROR: required command not found: %s\n' "$1" >&2
        exit 127
    }
}

render_template() {
    local template="$1" subject="$2"
    template="${template//\{base\}/$BASE_DIR}"
    printf '%s\n' "${template//\{subject\}/$subject}"
}

require_command flirt
require_command convert_xfm
require_command avscale
require_command fslval
require_command fslroi

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    for candidate in "$BASE_DIR"/*; do
        [[ -d "$candidate" ]] || continue
        subject=$(basename "$candidate")
        [[ "$subject" =~ ^[0-9]+$ ]] && SUBJECTS+=("$subject")
    done
fi

if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    printf 'ERROR: no numeric subject directories found under BASE_DIR: %s\n' "$BASE_DIR" >&2
    exit 2
fi

for subject in "${SUBJECTS[@]}"; do
    dwi=$(render_template "$DWI_TEMPLATE" "$subject")
    fmri=$(render_template "$FMRI_TEMPLATE" "$subject")
    outdir="$BASE_DIR/$subject/$OUTPUT_SUBDIR"

    printf '\n[%s]\n' "$subject"
    if [[ ! -f "$dwi" || ! -f "$fmri" ]]; then
        printf '  SKIP: missing DWI or mean-fMRI reference\n' >&2
        continue
    fi
    mkdir -p "$outdir"

    dwi_ref="$dwi"
    if [[ "$(fslval "$dwi" dim4)" -gt 1 ]]; then
        dwi_ref="$outdir/dwi_reference_volume0.nii.gz"
        fslroi "$dwi" "$dwi_ref" 0 1
    fi

    header_mat="$outdir/dwi_header_to_fmri.mat"
    estimated_mat="$outdir/dwi_flirt6_to_fmri.mat"
    header_inverse="$outdir/dwi_header_to_fmri_inverse.mat"
    residual_mat="$outdir/dwi_flirt6_vs_header.mat"

    flirt -in "$dwi_ref" -ref "$fmri" -usesqform -applyxfm \
        -omat "$header_mat" -out "$outdir/dwi_header_in_fmri.nii.gz"
    flirt -in "$dwi_ref" -ref "$fmri" -dof 6 -cost normmi -interp trilinear \
        -omat "$estimated_mat" -out "$outdir/dwi_flirt6_in_fmri.nii.gz"

    convert_xfm -omat "$header_inverse" -inverse "$header_mat"
    convert_xfm -omat "$residual_mat" -concat "$estimated_mat" "$header_inverse"
    avscale "$residual_mat" > "$outdir/dwi_flirt6_vs_header.txt"

    printf '  QC written: %s\n' "$outdir"
done
