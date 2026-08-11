%% =========================================================
%  gm_no3_denoise_v2.m
%  ---------------------------------------------------------
%  PhysIO noise modelling -> SPM 1st-level specification -> estimation
%
%  RESTORED FROM THE ORIGINAL PIPELINE
%    denoise_script_modelestimation_part_for_NTUSEC.m
%
%  The v1 rewrite (gm_no3_denoise.m) used only the six realignment
%  parameters as nuisance regressors.  The original used the PhysIO
%  toolbox to build a much richer nuisance model:
%      - noise ROI (CompCor-style) principal components from the SPM
%        tissue segmentations
%      - an expanded motion model (order 12 = 6 parameters + their
%        temporal derivatives)
%  Neither requires physiological recordings; the original left the
%  cardiac and respiratory log fields empty as well.  Dropping them was
%  an unintended loss during the rewrite.
%
%  INPUT   {BASE_DIR}/{subj}/gm/func/preprocessing/abr{subj}_rest.nii
%          {BASE_DIR}/{subj}/gm/func/preprocessing/rp_{subj}_rest.txt
%          {BASE_DIR}/{subj}/gm/anat/segmentation/c*{subj}_t1.nii
%  OUTPUT  {BASE_DIR}/{subj}/gm/func/denoise/multiple_regressors.txt
%          {BASE_DIR}/{subj}/gm/func/model/1st_level/SPM.mat, Res_*.nii
%
%  NEXT    gm_no4_detrend_bandpass_v2.m
%% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch';
SUBJECTS  = {'1123','1170'};   % subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/opt/spm/';

N_VOLS    = 240;      % number of volumes (check your DICOM)
TR        = 2;        % repetition time, seconds
N_SLICES  = 32;       % must equal the third image dimension
ONSET_SLICE = 2;      % reference slice used in slice-timing (ST_REFSLICE in no1_2)

% --- Noise ROIs for CompCor -------------------------------------------
%  NOISE_ROI_TISSUES selects which SPM tissue classes become noise ROIs.
%
%  [2 3]     white matter + CSF.  This is standard aCompCor
%            (Behzadi et al. 2007, NeuroImage 37:90) and is the DEFAULT
%            here.
%  [1 2 3 4 5]  what the original NTUSEC script used.  Including c1 makes
%            grey matter itself a noise ROI, so the leading principal
%            components of the grey-matter signal are regressed out.
%            That behaves much like global signal regression, which is
%            known to introduce spurious anti-correlations
%            (Murphy et al. 2009, NeuroImage 44:893) and would remove
%            part of the very signal this project measures.
%
%  Set to [1 2 3 4 5] if you want to reproduce the original exactly.
NOISE_ROI_TISSUES = [2 3];
NOISE_ROI_THRESHOLD = 0.95;   % tissue probability threshold for the ROI
NOISE_N_COMPONENTS  = 5;      % principal components per ROI

% --- Motion model -----------------------------------------------------
%  6  : realignment parameters only            (what v1 did)
%  12 : parameters + temporal derivatives      (what the original did)
%  24 : Friston-24 expansion
MOVEMENT_ORDER = 12;

% --- SPM implicit masking --------------------------------------------
%  mthresh is the implicit mask threshold: voxels whose mean signal falls
%  below MTHRESH * global mean are excluded from the model, and their
%  residuals are written as NaN.  Those voxels are therefore lost to every
%  downstream step.
%
%  0.8 is the SPM default for task fMRI and is what both the original and
%  v1 used.  It is restrictive for a whole-grey-matter resting-state
%  analysis.  Lower it to widen coverage; check n_gm after gm_no6 to see
%  the effect.
MTHRESH = 0.8;

HPF = 128;            % high-pass filter, seconds
%% =================================================

addpath(SPM_ROOT);
spm('defaults', 'FMRI');
spm_jobman('initcfg');

failed_subjects = {};

for s = 1:numel(SUBJECTS)
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, numel(SUBJECTS), subj);

    preproc_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'preprocessing');
    denoise_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'denoise');
    model_dir   = fullfile(BASE_DIR, subj, 'gm', 'func', 'model', '1st_level');
    seg_dir     = fullfile(BASE_DIR, subj, 'gm', 'anat', 'segmentation');

    if ~exist(denoise_dir, 'dir'), mkdir(denoise_dir); end
    if ~exist(model_dir,   'dir'), mkdir(model_dir);   end

    try
        % ---- locate inputs -------------------------------------------
        func_file = fullfile(preproc_dir, ['abr' subj '_rest.nii']);
        if ~exist(func_file, 'file')
            error('Preprocessed functional not found: %s', func_file);
        end

        rp_file = fullfile(preproc_dir, ['rp_' subj '_rest.txt']);
        if ~exist(rp_file, 'file')
            error('Realignment parameter file not found: %s', rp_file);
        end

        roi_files = cell(numel(NOISE_ROI_TISSUES), 1);
        for t = 1:numel(NOISE_ROI_TISSUES)
            k  = NOISE_ROI_TISSUES(t);
            dd = dir(fullfile(seg_dir, sprintf('c%d%s*.nii', k, subj)));
            if isempty(dd)
                error('Tissue segmentation c%d not found in %s', k, seg_dir);
            end
            roi_files{t} = fullfile(dd(1).folder, dd(1).name);
            fprintf('  noise ROI c%d : %s\n', k, dd(1).name);
        end

        clear matlabbatch;

        % ---------------------------------------------------------
        % Job 1: PhysIO - noise ROI components + expanded motion model
        % ---------------------------------------------------------
        matlabbatch{1}.spm.tools.physio.save_dir = {denoise_dir};

        % No physiological recordings; these fields stay empty, exactly as
        % in the original script.  PhysIO is used here purely for the noise
        % ROI and movement models.
        matlabbatch{1}.spm.tools.physio.log_files.vendor                   = 'BIDS';
        matlabbatch{1}.spm.tools.physio.log_files.cardiac                  = {''};
        matlabbatch{1}.spm.tools.physio.log_files.respiration              = {''};
        matlabbatch{1}.spm.tools.physio.log_files.scan_timing              = {''};
        matlabbatch{1}.spm.tools.physio.log_files.sampling_interval        = [];
        matlabbatch{1}.spm.tools.physio.log_files.relative_start_acquisition = 0;
        matlabbatch{1}.spm.tools.physio.log_files.align_scan               = 'last';

        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices     = N_SLICES;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR          = TR;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies    = 0;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans      = N_VOLS;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = ONSET_SLICE;

        matlabbatch{1}.spm.tools.physio.preproc.cardiac.modality = 'ECG';
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 90;
        matlabbatch{1}.spm.tools.physio.preproc.respiratory.filter.passband = [0.01 2];

        matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors = 'multiple_regressors.txt';
        matlabbatch{1}.spm.tools.physio.model.output_physio              = 'physio.mat';

        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.fmri_files      = {func_file};
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.roi_files       = roi_files;
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.force_coregister = 'Yes';
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.thresholds      = NOISE_ROI_THRESHOLD;
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.n_components    = NOISE_N_COMPONENTS;

        matlabbatch{1}.spm.tools.physio.model.movement.yes.file_realignment_parameters = {rp_file};
        matlabbatch{1}.spm.tools.physio.model.movement.yes.order = MOVEMENT_ORDER;

        matlabbatch{1}.spm.tools.physio.verbose.level = 1;

        % ---------------------------------------------------------
        % Job 2: fMRI model specification
        % ---------------------------------------------------------
        scans = cell(N_VOLS, 1);
        for v = 1:N_VOLS
            scans{v} = sprintf('%s,%d', func_file, v);
        end

        matlabbatch{2}.spm.stats.fmri_spec.dir            = {model_dir};
        matlabbatch{2}.spm.stats.fmri_spec.timing.units   = 'secs';
        matlabbatch{2}.spm.stats.fmri_spec.timing.RT      = TR;
        matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t  = N_SLICES;   % must match Nslices
        matlabbatch{2}.spm.stats.fmri_spec.timing.fmri_t0 = ONSET_SLICE;
        matlabbatch{2}.spm.stats.fmri_spec.sess.scans     = scans;
        matlabbatch{2}.spm.stats.fmri_spec.sess.cond      = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
        matlabbatch{2}.spm.stats.fmri_spec.sess.multi     = {''};
        matlabbatch{2}.spm.stats.fmri_spec.sess.regress   = struct('name', {}, 'val', {});
        matlabbatch{2}.spm.stats.fmri_spec.sess.multi_reg = {fullfile(denoise_dir, 'multiple_regressors.txt')};
        matlabbatch{2}.spm.stats.fmri_spec.sess.hpf       = HPF;
        matlabbatch{2}.spm.stats.fmri_spec.fact           = struct('name', {}, 'levels', {});
        matlabbatch{2}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{2}.spm.stats.fmri_spec.volt           = 1;
        matlabbatch{2}.spm.stats.fmri_spec.global         = 'None';
        matlabbatch{2}.spm.stats.fmri_spec.mthresh        = MTHRESH;
        matlabbatch{2}.spm.stats.fmri_spec.mask           = {''};
        matlabbatch{2}.spm.stats.fmri_spec.cvi            = 'AR(1)';

        % ---------------------------------------------------------
        % Job 3: model estimation, write residuals
        % ---------------------------------------------------------
        matlabbatch{3}.spm.stats.fmri_est.spmmat           = {fullfile(model_dir, 'SPM.mat')};
        matlabbatch{3}.spm.stats.fmri_est.write_residuals  = 1;
        matlabbatch{3}.spm.stats.fmri_est.method.Classical = 1;

        spm_jobman('run', matlabbatch);

        % ---- report the nuisance model actually used ------------------
        mr = fullfile(denoise_dir, 'multiple_regressors.txt');
        if exist(mr, 'file')
            R = load(mr);
            fprintf('  nuisance regressors written: %d columns (%d volumes)\n', ...
                    size(R, 2), size(R, 1));
        end

        nres = numel(dir(fullfile(model_dir, 'Res_*.nii')));
        fprintf('  residual volumes written: %d (expected %d)\n', nres, N_VOLS);
        if nres ~= N_VOLS
            warning('Residual count does not match N_VOLS for %s', subj);
        end

        fprintf('[OK] %s\n', subj);

    catch err
        fprintf('[FAILED] %s\n', subj);
        fprintf('  Error: %s\n', err.message);
        failed_subjects{end+1} = subj; %#ok<SAGROW>
    end

    clear matlabbatch;
end

%% ================= SUMMARY =================
fprintf('\n=========================================\n');
fprintf('Done: %d / %d subjects succeeded\n', ...
        numel(SUBJECTS) - numel(failed_subjects), numel(SUBJECTS));
if isempty(failed_subjects)
    fprintf('All subjects completed successfully!\n');
else
    fprintf('Failed subjects (%d):\n', numel(failed_subjects));
    for i = 1:numel(failed_subjects)
        fprintf('  - %s\n', failed_subjects{i});
    end
end
fprintf('=========================================\n');
