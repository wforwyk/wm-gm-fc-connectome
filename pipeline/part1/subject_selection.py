"""Participant-list loader for standalone Part 1 scripts."""
from pathlib import Path

def resolve_subjects(subjects, subjects_txt, exclude_subjects, batch_dir):
    if subjects_txt:
        subjects = [x.strip() for x in Path(subjects_txt).read_text().splitlines()
                    if x.strip() and not x.lstrip().startswith('#')]
        if not subjects:
            raise ValueError('SUBJECTS_TXT contains no participant IDs.')
    if not subjects:
        subjects = sorted(p.name for p in Path(batch_dir).iterdir() if p.is_dir() and p.name.isdigit())
    excluded = {str(x).removeprefix('sub-') for x in exclude_subjects}
    selected = sorted({str(x).removeprefix('sub-') for x in subjects} - excluded)
    if any(not x.isdigit() for x in selected):
        raise ValueError('Subject IDs must be numeric.')
    print(f'N subjects: {len(selected)}; excluded: {sorted(excluded)}')
    return selected
