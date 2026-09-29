"""Consistent participant selection for Part 2; atlas-only stages do not use it."""
from pathlib import Path

# USER SETTINGS: one shared cohort for every participant/group stage.
SUBJECTS = []                 # Numeric derivative IDs; empty discovers available inputs.
SUBJECTS_TXT = ''             # One numeric ID per line; overrides SUBJECTS when set.
EXCLUDE_SUBJECTS = []          # Participant exclusions; apply at every stage.


def select_subjects(available):
    requested = SUBJECTS
    if SUBJECTS_TXT:
        requested = [line.strip() for line in Path(SUBJECTS_TXT).read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
        if not requested:
            raise ValueError('SUBJECTS_TXT contains no participant IDs.')
    excluded = {str(x).removeprefix('sub-') for x in EXCLUDE_SUBJECTS}
    candidates = [str(x).removeprefix('sub-') for x in (requested or available)]
    if any(not x.isdigit() for x in candidates):
        raise ValueError('Subject IDs must be numeric (optional sub- prefix).')
    selected = sorted(set(candidates) - excluded)
    print(f'N subjects: {len(selected)}; excluded: {sorted(excluded)}')
    return selected


def select_files(files):
    files = list(files)
    selected = set(select_subjects([Path(f).name.split('_')[0] for f in files]))
    return [f for f in files if Path(f).name.split('_')[0] in selected]


def filter_subject_rows(frame):
    if 'subject' not in frame:
        raise ValueError('Participant table lacks subject column.')
    keys = frame.subject.astype(str).str.removeprefix('sub-')
    selected = set(select_subjects(keys.unique().tolist()))
    # CSV readers may infer numeric IDs and remove leading zeros.
    canonical = lambda x: str(int(x))
    selected = {canonical(x) for x in selected}
    return frame.loc[keys.map(canonical).isin(selected)].copy()
