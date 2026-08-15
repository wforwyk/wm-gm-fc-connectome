#!/bin/bash

# ==============================================================================
# USER SETTINGS
# ==============================================================================
PROJECT_DIR="/path/to/your/project"
RAW_DIR="${PROJECT_DIR}/rawdata"
BATCH_DIR="${PROJECT_DIR}/derivatives/batch"
RAW_SESSION="ses-01"  # Must match pt1_gm_no00_setting_directory_data.sh.

SUBJECT_LIST=("0001" "0002") # Numeric derivative IDs; replace with yours.
# ==============================================================================

echo "=============================================================================="
echo "Starting FreeSurfer Preprocessing Pipeline"
echo "PROJECT_DIR Path: ${PROJECT_DIR}"
echo "Total subjects to process: ${#SUBJECT_LIST[@]}"
echo "=============================================================================="

failed_subjects=()

for SUBJ in "${SUBJECT_LIST[@]}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[ Processing Subject: ${SUBJ} ]"
    echo "------------------------------------------------------------------"

    FS_DIR="${BATCH_DIR}/${SUBJ}/wm/freesurfer"
    export SUBJECTS_DIR="${FS_DIR}"
    T1_FILE="${RAW_DIR}/sub-${SUBJ}/${RAW_SESSION}/anat/sub-${SUBJ}_t1.nii"

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
