#!/bin/bash
# ============================================================
# no0_copy_u_rc1.sh
# -----------------------------------------------------------
# PURPOSE : Copy each subject's DARTEL flow field into the batch derivatives.
#           from the DARTEL source directory into each subject's
#           gm/derived/ folder in the batch derivatives.
#
# INPUT   : {FLOW_FIELD_SOURCE_DIR}/sub-{subject}/{SOURCE_SESSION}/anat/
#               u_rc1sub-{subject}_{SOURCE_SESSION}_T1w_{DARTEL_TEMPLATE_ID}.nii
# OUTPUT  : {BATCH_DIR}/{subject}/gm/derived/
#
# NOTE    : Source folder uses sub- prefix; batch folders use numeric IDs.
#           Edit SUBJECTS (numeric IDs only) in USER SETTINGS.
#           Already-existing files are skipped (not overwritten).
# ============================================================

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS=("0001" "0002")  # Numeric derivative IDs; replace with yours.
FLOW_FIELD_SOURCE_DIR="/path/to/your/dartel_flow_field_source"
BATCH_DIR="/path/to/your/project/derivatives/batch"
SOURCE_SESSION="ses-01"
DARTEL_TEMPLATE_ID="YOUR_DARTEL_TEMPLATE"
# ────────────────────────────────────────────────────────────

echo "============================================"
echo "  step1_copy_u_rc1.sh"
echo "  Subjects : ${#SUBJECTS[@]}"
echo "============================================"

FAIL=()

for subj in "${SUBJECTS[@]}"; do
    fname="u_rc1sub-${subj}_${SOURCE_SESSION}_T1w_${DARTEL_TEMPLATE_ID}.nii"
    src="${FLOW_FIELD_SOURCE_DIR}/sub-${subj}/${SOURCE_SESSION}/anat/${fname}"
    dest_dir="${BATCH_DIR}/${subj}/gm/derived"
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
