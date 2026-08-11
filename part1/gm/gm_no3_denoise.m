% =========================================================
% no3_denoise.m
% SPM fMRI 1st-level model specification & estimation

% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/your/base/directory/is/your/root/directory';
SUBJECTS  = {'1001','1002'};  # subject_id must be in ' ' and divided by ,


N_VOLS    = 240; # check your dicom. 
SPM_ROOT  = '/opt/spm/';

%% ================= SPM INITIALIZATION =================
addpath(SPM_ROOT);
spm('defaults', 'FMRI');
spm_jobman('initcfg');

%% ================= MAIN LOOP =================
failed_subjects = {};

for s = 1:length(SUBJECTS)
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, length(SUBJECTS), subj);

    preproc_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'preprocessing');
    model_dir   = fullfile(BASE_DIR, subj, 'gm', 'func', 'model', '1st_level');

    % create output directory if it does not exist
    if ~exist(model_dir, 'dir')
        mkdir(model_dir);
    end

    % build scans list (one entry per volume)
    nii_file = fullfile(preproc_dir, ['abr' subj '_rest.nii']);
    scans = cell(N_VOLS, 1);
    for v = 1:N_VOLS
        scans{v} = [nii_file ',' num2str(v)];
    end

    % noise regressors file
    multi_reg = fullfile(preproc_dir, ['rp_' subj '_rest.txt']);

    % ---------------------------------------------------------
    % Job 1: fMRI model specification
    % ---------------------------------------------------------
    matlabbatch{1}.spm.stats.fmri_spec.dir              = {model_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units     = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT        = 2;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t    = 34;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0   = 2;
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans       = scans;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond        = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi       = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.regress     = struct('name', {}, 'val', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg   = {multi_reg};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf         = 128;
    matlabbatch{1}.spm.stats.fmri_spec.fact             = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt             = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global           = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh          = 0.8;
    matlabbatch{1}.spm.stats.fmri_spec.mask             = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi              = 'AR(1)';

    % ---------------------------------------------------------
    % Job 2: fMRI model estimation
    % ---------------------------------------------------------
    matlabbatch{2}.spm.stats.fmri_est.spmmat           = {fullfile(model_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals  = 1;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    % ---------------------------------------------------------
    % Run (skip subject on error and continue)
    % ---------------------------------------------------------
    try
        spm_jobman('run', matlabbatch);
        fprintf('[OK] %s\n', subj);
    catch err
        fprintf('[FAILED] %s\n', subj);
        fprintf('  Error: %s\n', err.message);
        failed_subjects{end+1} = subj;
    end
    clear matlabbatch;

end  % main loop

%% ================= SUMMARY =================
fprintf('\n=========================================\n');
fprintf('Done: %d / %d subjects succeeded\n', length(SUBJECTS) - length(failed_subjects), length(SUBJECTS));
if isempty(failed_subjects)
    fprintf('All subjects completed successfully!\n');
else
    fprintf('Failed subjects (%d):\n', length(failed_subjects));
    for i = 1:length(failed_subjects)
        fprintf('  - %s\n', failed_subjects{i});
    end
end
fprintf('=========================================\n');
