#!/bin/bash

# ==============================================================================
# USER SETTINGS
# ==============================================================================
BASE_DIR="/your/designated/base/directory"
RAW_DIR="${BASE_DIR}/rawdata"
DERIV_DIR="${BASE_DIR}/derivatives"

SUBJECT_LIST=("1001" "1002") # e.g. subject id must be in " " and divided by space
# ==============================================================================

echo "=============================================================================="
echo "Starting FreeSurfer Preprocessing Pipeline"
echo "BASE_DIR Path: ${BASE_DIR}"
echo "Total subjects to process: ${#SUBJECT_LIST[@]}"
echo "=============================================================================="

failed_subjects=()

for SUBJ in "${SUBJECT_LIST[@]}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[ Processing Subject: ${SUBJ} ]"
    echo "------------------------------------------------------------------"

    FS_DIR="${DERIV_DIR}/${SUBJ}/wm/freesurfer"
    export SUBJECTS_DIR="${FS_DIR}"
    T1_FILE="${RAW_DIR}/${SUBJ}/ses001/anat/${SUBJ}_t1.nii"

    echo "T1 Input File: ${T1_FILE}"
    echo "Output Directory: ${FS_DIR}/${SUBJ}"

    # Step 1: recon-all
    echo ">>> Step 1: Running recon-all for ${SUBJ} (This will take 3~3.5 hours)..."
    recon-all -s "${SUBJ}" -i "${T1_FILE}" -all

    if [ $? -ne 0 ]; then
        echo "[FAILED] recon-all failed for ${SUBJ}. Skipping."
        failed_subjects+=("${SUBJ}")
        continue
    fi

    # Step 2: segmentThalamicNuclei.sh
    echo ">>> Step 2: Running segmentThalamicNuclei.sh for ${SUBJ}..."
    segmentThalamicNuclei.sh "${SUBJ}"

    if [ $? -ne 0 ]; then
        echo "[FAILED] segmentThalamicNuclei.sh failed for ${SUBJ}."
        failed_subjects+=("${SUBJ}")
        continue
    fi

    echo "[OK] ${SUBJ} finished successfully."

done

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=============================================================================="
total=${#SUBJECT_LIST[@]}
n_failed=${#failed_subjects[@]}
n_ok=$(( total - n_failed ))
echo "Done: ${n_ok} / ${total} subjects succeeded"
if [[ ${n_failed} -eq 0 ]]; then
    echo "All subjects completed successfully!"
else
    echo "Failed subjects (${n_failed}):"
    for s in "${failed_subjects[@]}"; do
        echo "  - ${s}"
    done
fi
echo "=============================================================================="