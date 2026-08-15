#!/usr/bin/env bash
set -uo pipefail

# =========================
# USER SETTINGS
# =========================
PROJECT_DIR="/path/to/your/project"
SOURCE_DIR="/path/to/your/source_data"

# Inspect your source-data layout before running.  This example assumes
# BIDS-like raw folders (sub-0001/ses-01) and conventional T1/rest names.
SOURCE_SESSION="ses-01"
RAW_SESSION="ses-01"
SUBJECT_IDS=("0001" "0002")  # Four-digit numeric IDs; replace with yours.

# =========================

echo "PROJECT_DIR : $PROJECT_DIR"
echo "SOURCE_DIR  : $SOURCE_DIR"

failed_subjects=()

for ID in "${SUBJECT_IDS[@]}"; do
  RAW_SUBJ="sub-${ID}"
  echo "--------------------------------------"
  echo "Processing ${RAW_SUBJ} -> ${ID}"

  subj_failed=0

  # ---------- STEP 1: source -> rawdata ---------- # this can be edited or removed as needed
  RAW_PATH="${PROJECT_DIR}/rawdata/${RAW_SUBJ}/${RAW_SESSION}"
  mkdir -p "${RAW_PATH}/anat" "${RAW_PATH}/func"

  T1_SRC="${SOURCE_DIR}/${RAW_SUBJ}/${SOURCE_SESSION}/anat/${RAW_SUBJ}_${SOURCE_SESSION}_T1w.nii"
  REST_SRC="${SOURCE_DIR}/${RAW_SUBJ}/${SOURCE_SESSION}/func/${RAW_SUBJ}_${SOURCE_SESSION}_task-rest_bold.nii"

  T1_RAW="${RAW_PATH}/anat/${RAW_SUBJ}_t1.nii"
  REST_RAW="${RAW_PATH}/func/${RAW_SUBJ}_rest.nii"

  if [[ -f "${T1_SRC}" ]]; then
    cp -n "${T1_SRC}" "${T1_RAW}"
    echo "[OK] T1 -> rawdata"
  else
    echo "[MISS] T1 source not found: ${T1_SRC}"
    subj_failed=1
  fi

  if [[ -f "${REST_SRC}" ]]; then
    cp -n "${REST_SRC}" "${REST_RAW}"
    echo "[OK] REST -> rawdata"
  else
    echo "[MISS] REST source not found: ${REST_SRC}"
    subj_failed=1
  fi

  # ---------- STEP 2: create derivatives folder structure ----------
  # Raw data retain the BIDS-style sub- prefix.  The batch derivatives use
  # the numeric ID only, which is the convention expected by later scripts.
  DER_SUB="${PROJECT_DIR}/derivatives/batch/${ID}"

  # GM
  GM="${DER_SUB}/gm"
  mkdir -p \
    "${GM}/anat/segmentation" \
    "${GM}/anat/qc" \
    "${GM}/func/preprocessing" \
    "${GM}/func/preprocessing/qc" \
    "${GM}/func/model/1st_level" \
    "${GM}/func/derived/fc_results" \
    "${GM}/func/derived/distance/path" \


  # WM
  mkdir -p \
    "${DER_SUB}/wm/freesurfer" \
    "${DER_SUB}/wm/tracula" \
    "${DER_SUB}/wm/derived"

  # ---------- STEP 3: rawdata -> derivatives ----------
  T1_WORK_DIR="${GM}/anat/segmentation"
  REST_WORK_DIR="${GM}/func/preprocessing"

  if [[ -f "${T1_RAW}" ]]; then
    cp -n "${T1_RAW}" "${T1_WORK_DIR}/${ID}_t1.nii"
    echo "[OK] T1 -> derivatives gm/anat/segmentation"
  else
    echo "[MISS] T1 rawdata not found: ${T1_RAW}"
    subj_failed=1
  fi

  if [[ -f "${REST_RAW}" ]]; then
    cp -n "${REST_RAW}" "${REST_WORK_DIR}/${ID}_rest.nii"
    echo "[OK] REST -> derivatives gm/func/preprocessing"
  else
    echo "[MISS] REST rawdata not found: ${REST_RAW}"
    subj_failed=1
  fi

  if [[ $subj_failed -eq 1 ]]; then
    failed_subjects+=("${SUBJ}")
  fi

done

# =========================
# SUMMARY
# =========================
echo ""
echo "========================================="
total=${#SUBJECT_IDS[@]}
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
echo "========================================="
