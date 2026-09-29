#!/usr/bin/env bash
# Source after USER SETTINGS. Empty SUBJECTS discovers numeric derivative directories.
subject_selection_nounset=false
[[ $- == *u* ]] && subject_selection_nounset=true
set +u
if [[ -n "$SUBJECTS_TXT" ]]; then
    [[ -f "$SUBJECTS_TXT" ]] || { echo "Missing SUBJECTS_TXT: $SUBJECTS_TXT" >&2; exit 2; }
    SUBJECTS=()
    while IFS= read -r subject || [[ -n "$subject" ]]; do
        subject="${subject%$'\r'}"
        subject="${subject#"${subject%%[![:space:]]*}"}"
        subject="${subject%"${subject##*[![:space:]]}"}"
        [[ -z "$subject" || "$subject" == \#* ]] && continue
        SUBJECTS+=("${subject#sub-}")
    done < "$SUBJECTS_TXT"
    [[ ${#SUBJECTS[@]} -gt 0 ]] || { echo "SUBJECTS_TXT contains no participant IDs." >&2; exit 2; }
fi
if [[ ${#SUBJECTS[@]} -eq 0 ]]; then
    for candidate in "$BATCH_DIR"/*; do
        [[ -d "$candidate" ]] || continue
        subject="${candidate##*/}"
        [[ "$subject" =~ ^[0-9]+$ ]] && SUBJECTS+=("$subject")
    done
fi
selected_subjects=()
for subject in "${SUBJECTS[@]}"; do
    subject="${subject#sub-}"
    [[ "$subject" =~ ^[0-9]+$ ]] || { echo "Invalid subject ID: $subject" >&2; exit 2; }
    excluded=false
    for exclusion in "${EXCLUDE_SUBJECTS[@]}"; do
        [[ "$subject" == "${exclusion#sub-}" ]] && excluded=true
    done
    $excluded || selected_subjects+=("$subject")
done
SUBJECTS=("${selected_subjects[@]}")
printf 'N subjects: %s; excluded: %s\n' "${#SUBJECTS[@]}" "${EXCLUDE_SUBJECTS[*]}"
[[ ${#SUBJECTS[@]} -gt 0 ]] || { echo 'No subjects selected.' >&2; exit 2; }

if $subject_selection_nounset; then set -u; fi
