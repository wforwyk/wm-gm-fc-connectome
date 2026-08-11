#!/bin/bash
# ============================================================
# no0_copy_u_rc1.sh
# -----------------------------------------------------------
# PURPOSE : Copy DARTEL flow field u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
#           from the DARTEL source directory into each subject's
#           gm/derived/ folder in the batch2 pipeline.
#
# INPUT   : {SRC_DIR}/sub-{subject}/ses-1/anat/
#               u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
# OUTPUT  : {TEMPLATE_DIR}/{subject}/gm/derived/
#               u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
#
# NOTE    : Source folder uses sub- prefix; batch2 folders do not.
#           Edit SUBJECTS (numeric IDs only) in USER SETTINGS.
#           Already-existing files are skipped (not overwritten).
# ============================================================

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS=("1001" "1002") # e.g. subject id must be in " " and divided by space

SRC_DIR="/bml/projects/06_resilience/projects/06-04_resilience-brain-state-transitions/projects/NCKU/data/derivatives/DARTEL_336Ss/NCKU_189Ss"
TEMPLATE_DIR="/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch2/derivatives"
# ────────────────────────────────────────────────────────────

echo "============================================"
echo "  step1_copy_u_rc1.sh"
echo "  Subjects : ${#SUBJECTS[@]}"
echo "============================================"

FAIL=()

for subj in "${SUBJECTS[@]}"; do
    fname="u_rc1sub-${subj}_ses-1_T1w_NCKU_336Ss.nii"
    src="${SRC_DIR}/sub-${subj}/ses-1/anat/${fname}"
    dest_dir="${TEMPLATE_DIR}/${subj}/gm/derived"
    dest="${dest_dir}/${fname}"

    echo ""
    echo "[${subj}]"

    if [ ! -f "$src" ]; then
        echo "  ERROR: source not found: $src"
        FAIL+=("$subj")
        continue
    fi

    mkdir -p "$dest_dir"

    if [ -f "$dest" ]; then
        echo "  SKIP : already exists: $dest"
        continue
    fi

    cp "$src" "$dest"
    if [ $? -eq 0 ]; then
        echo "  OK   : copied -> $dest"
    else
        echo "  ERROR: copy failed"
        FAIL+=("$subj")
    fi
done

echo ""
echo "============================================"
if [ ${#FAIL[@]} -eq 0 ]; then
    echo "  All done. No failures."
else
    echo "  FAILED subjects (${#FAIL[@]}):"
    for f in "${FAIL[@]}"; do echo "    $f"; done
fi
echo "============================================"