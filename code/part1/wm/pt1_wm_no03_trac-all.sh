#!/bin/bash

# ==============================================================================
# USER SETTINGS
# ==============================================================================
ANTSPATH="/path/to/your/ants/bin"
BATCH_DIR="/path/to/your/project/derivatives/batch"

SUBJECTS=("0001" "0002") # e.g. subject id must be in " " and divided by space

# ==============================================================================

export ANTSPATH
export PATH="${ANTSPATH}:$PATH"

failed_subjects=()

for subj in "${SUBJECTS[@]}"; do
    echo "============================================================"
    echo "Starting processing for ${subj}"
    echo "============================================================"

    config_file="${BATCH_DIR}/${subj}/wm/tracula/${subj}_dmrirc"

    # Step 1: -prep
    echo "[${subj}] Running trac-all -prep ..."
    trac-all -prep -c "$config_file"
    if [ $? -ne 0 ]; then
        echo "[FAILED] ${subj}: trac-all -prep failed. Skipping subject."
        failed_subjects+=("${subj}")
        continue
    fi

    # Step 2: -bedp
    echo "[${subj}] Running trac-all -bedp ..."
    trac-all -bedp -c "$config_file"
    if [ $? -ne 0 ]; then
        echo "[FAILED] ${subj}: trac-all -bedp failed. Skipping subject."
        failed_subjects+=("${subj}")
        continue
    fi

    # Step 3: -path
    echo "[${subj}] Running trac-all -path ..."
    trac-all -path -c "$config_file"
    if [ $? -ne 0 ]; then
        echo "[FAILED] ${subj}: trac-all -path failed. Skipping subject."
        failed_subjects+=("${subj}")
        continue
    fi

    echo "[OK] ${subj}: -prep, -bedp, -path all finished successfully."
    echo ""

done

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "============================================================"
total=${#SUBJECTS[@]}
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
echo "============================================================"
