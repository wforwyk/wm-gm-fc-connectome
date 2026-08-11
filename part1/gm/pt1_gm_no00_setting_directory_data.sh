#!/usr/bin/env bash
set -uo pipefail

# =========================
# USER SETTINGS
# =========================
BASE_DIR="/your/designated/base/directory"
SRC_DIR="/where/your/sourcedata/are/saved/and_to_be/moved/to/your/BASE_DIR/directory"

SES_SRC="ses-1" # this can be edited or removed as needed
SES_DST="ses001" # this can be edited or removed as needed

SUBJECTS=("1001" "1002") # e.g. subject id must be in " " and divided by space

# =========================

echo "BASE_DIR     : $BASE_DIR"
echo "SRC_DIR : $SRC_DIR"

failed_subjects=()

for ID in "${SUBJECTS[@]}"; do
  SUBJ="sub-${ID}"
  echo "--------------------------------------"
  echo "Processing ${SUBJ}"

  subj_failed=0

  # ---------- STEP 1: source -> rawdata ---------- # this can be edited or removed as needed
  RAW_PATH="${BASE_DIR}/rawdata/${SUBJ}/${SES_DST}"
  mkdir -p "${RAW_PATH}/anat" "${RAW_PATH}/func"

  T1_SRC="${SRC_DIR}/${SUBJ}/${SES_SRC}/anat/${SUBJ}_${SES_SRC}_T1w.nii"
  REST_SRC="${SRC_DIR}/${SUBJ}/${SES_SRC}/func/${SUBJ}_${SES_SRC}_task-rest_bold.nii"

  T1_RAW="${RAW_PATH}/anat/${SUBJ}_t1.nii"
  REST_RAW="${RAW_PATH}/func/${SUBJ}_rest.nii"

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
  DER_SUB="${BASE_DIR}/derivatives/${SUBJ}"

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
    cp -n "${T1_RAW}" "${T1_WORK_DIR}/"
    echo "[OK] T1 -> derivatives gm/anat/segmentation"
  else
    echo "[MISS] T1 rawdata not found: ${T1_RAW}"
    subj_failed=1
  fi

  if [[ -f "${REST_RAW}" ]]; then
    cp -n "${REST_RAW}" "${REST_WORK_DIR}/"
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
echo "========================================="