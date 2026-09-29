function subjects = resolve_participant_list(subjects, subjects_txt, exclude_subjects, batch_dir)
% Read one numeric ID per line; a nonempty file overrides the inline list.
if ~isempty(subjects_txt)
    lines = regexp(fileread(subjects_txt), '\r?\n', 'split');
    subjects = strtrim(lines);
    subjects = subjects(~cellfun(@isempty, subjects));
    subjects = subjects(~startsWith(subjects, '#'));
    if isempty(subjects), error('SUBJECTS_TXT contains no participant IDs.'); end
end
if isempty(subjects)
    entries = dir(batch_dir);
    subjects = {entries([entries.isdir]).name};
    subjects = subjects(~cellfun(@isempty, regexp(subjects, '^\d+$', 'once')));
end
subjects = regexprep(subjects, '^sub-', '');
exclude_subjects = regexprep(exclude_subjects, '^sub-', '');
if any(cellfun(@isempty, regexp(subjects, '^\d+$', 'once')))
    error('Subject IDs must be numeric.');
end
subjects = setdiff(subjects, exclude_subjects, 'stable');
fprintf('N subjects: %d; excluded: %s\n', numel(subjects), strjoin(exclude_subjects, ', '));
end
