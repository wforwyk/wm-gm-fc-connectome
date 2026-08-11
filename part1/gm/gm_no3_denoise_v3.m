%% =========================================================
%  gm_no3_denoise_v3.m
%  ---------------------------------------------------------
%  Nuisance regressors -> SPM 1st-level specification -> estimation
%
%  WHY v3 REPLACES v2
%    v2 called the TAPAS PhysIO toolbox, which failed with
%        Failed 'TAPAS PhysIO Toolbox'
%        Unrecognized field name "dt".
%        in tapas_physio_write2bids.m
%    Setting log_files.vendor = 'BIDS' makes PhysIO write the physiological
%    traces back out in BIDS format.  There are no physiological
%    recordings in this study, so that structure has no dt field and the
%    writer errors.  The original NTUSEC script used the same setting and
%    presumably ran on an older TAPAS release.
%
%    PhysIO was only ever being used for two things here:
%       (a) principal components of tissue-ROI signal  (CompCor)
%       (b) an expanded motion model
%    Both are a few lines of MATLAB.  Computing them directly removes the
%    toolbox dependency, removes this failure mode, and makes the nuisance
%    model explicit and inspectable.
%
%  NUISANCE MODEL
%    CompCor : first N principal components of the time series inside each
%              noise ROI, following Behzadi et al. (2007) NeuroImage 37:90
%              - tissue probability map resliced to the functional grid
%              - thresholded, optionally eroded to limit partial volume
%              - each voxel mean-and-trend removed, then variance
%                normalised, then SVD
%    Motion  : realignment parameters expanded to the requested order
%              6  = parameters
%              12 = parameters + temporal derivatives   (matches the
%                   original PhysIO movement order 12)
%              24 = Friston-24 (parameters, derivatives, and squares)
%
%  INPUT   {BASE_DIR}/{subj}/gm/func/preprocessing/abr{subj}_rest.nii
%          {BASE_DIR}/{subj}/gm/func/preprocessing/rp_{subj}_rest.txt
%          {BASE_DIR}/{subj}/gm/anat/segmentation/c*{subj}*.nii
%  OUTPUT  {BASE_DIR}/{subj}/gm/func/denoise/multiple_regressors.txt
%          {BASE_DIR}/{subj}/gm/func/denoise/nuisance_qc.txt
%          {BASE_DIR}/{subj}/gm/func/model/1st_level/SPM.mat, Res_*.nii
%
%  NEXT    gm_no4_detrend_bandpass_v2.m
%% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch';
SUBJECTS  = {'1123','1170'};   % subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/opt/spm/';

N_VOLS    = 240;      % number of volumes
TR        = 2;        % repetition time, seconds
N_SLICES  = 32;       % must equal the third image dimension
ONSET_SLICE = 2;      % reference slice used in slice timing (ST_REFSLICE in no1_2)

% --- CompCor noise ROIs ------------------------------------------------
%  Which SPM tissue classes become noise ROIs.
%
%  [2 3]        white matter + CSF.  Standard aCompCor, and the DEFAULT.
%  [1 2 3 4 5]  what the original NTUSEC script passed to PhysIO.  Adding
%               c1 makes grey matter a noise ROI, so the leading principal
%               components of grey-matter signal get regressed out.  That
%               behaves much like global signal regression, which induces
%               spurious anti-correlations (Murphy et al. 2009, NeuroImage
%               44:893) and would remove part of the signal this project
%               is measuring.  Set this only to reproduce the original.
NOISE_ROI_TISSUES   = [2 3];
NOISE_ROI_THRESHOLD = 0.95;   % tissue probability threshold
NOISE_N_COMPONENTS  = 5;      % principal components per ROI
NOISE_ROI_ERODE     = 1;      % erosion iterations; 0 disables

% --- Motion model ------------------------------------------------------
MOVEMENT_ORDER = 12;          % 6, 12, or 24

% --- SPM implicit masking ---------------------------------------------
%  Voxels whose mean signal is below MTHRESH * global mean are excluded
%  from the model and their residuals are written as NaN, so they are lost
%  to every downstream step.  0.8 is the SPM default and is what both the
%  original and v1 used; it is restrictive for whole-grey-matter resting
%  state.  Check n_gm in gm_no6_qc.csv before settling on a value.
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

        % ---- load the functional once --------------------------------
        Vf = spm_vol(func_file);
        if numel(Vf) ~= N_VOLS
            warning('%s has %d volumes, N_VOLS is %d. Using the file.', ...
                    subj, numel(Vf), N_VOLS);
        end
        nT = numel(Vf);
        Yf = spm_read_vols(Vf);
        dims = size(Yf);
        Y2 = reshape(Yf, [], nT)';                     % [nT x nVox]
        clear Yf;

        qc_lines = {};
        qc_lines{end+1} = sprintf('subject               : %s', subj); %#ok<SAGROW>
        qc_lines{end+1} = sprintf('volumes               : %d', nT);   %#ok<SAGROW>

        % ---- CompCor components --------------------------------------
        compcor = [];
        for t = 1:numel(NOISE_ROI_TISSUES)
            k  = NOISE_ROI_TISSUES(t);
            dd = dir(fullfile(seg_dir, sprintf('c%d%s*.nii', k, subj)));
            if isempty(dd)
                error('Tissue segmentation c%d not found in %s', k, seg_dir);
            end
            tissue_path = fullfile(dd(1).folder, dd(1).name);

            roi = reslice_prob_map(tissue_path, Vf(1), dims);
            roi_mask = roi > NOISE_ROI_THRESHOLD;

            for e = 1:NOISE_ROI_ERODE
                roi_mask = imerode_26(roi_mask);
            end

            nvox = nnz(roi_mask);
            if nvox < NOISE_N_COMPONENTS * 5
                warning(['Noise ROI c%d has only %d voxels after threshold ' ...
                         'and erosion; skipping.'], k, nvox);
                qc_lines{end+1} = sprintf('CompCor c%d           : SKIPPED (%d voxels)', k, nvox); %#ok<SAGROW>
                continue;
            end

            [comp, varexp] = compcor_components(Y2(:, roi_mask(:)), ...
                                                NOISE_N_COMPONENTS);
            compcor = [compcor, comp]; %#ok<AGROW>

            fprintf('  CompCor c%d: %d voxels, %d components, %.1f%% variance\n', ...
                    k, nvox, size(comp, 2), 100 * sum(varexp));
            qc_lines{end+1} = sprintf(['CompCor c%d           : %d voxels, ' ...
                'th %.2f, erode %d, %d comps, var explained %.1f%%'], ...
                k, nvox, NOISE_ROI_THRESHOLD, NOISE_ROI_ERODE, ...
                size(comp, 2), 100 * sum(varexp)); %#ok<SAGROW>
        end
        clear Y2;

        % ---- motion regressors ---------------------------------------
        rp = load(rp_file);
        if size(rp, 1) ~= nT
            error('rp file has %d rows, functional has %d volumes.', ...
                  size(rp, 1), nT);
        end
        motion = expand_motion(rp, MOVEMENT_ORDER);
        fprintf('  Motion: order %d -> %d regressors\n', ...
                MOVEMENT_ORDER, size(motion, 2));
        qc_lines{end+1} = sprintf('motion                : order %d, %d regressors', ...
                                  MOVEMENT_ORDER, size(motion, 2)); %#ok<SAGROW>

        % mean framewise displacement, 50 mm radius (Power et al. 2012)
        fd = mean([0; sum(abs(diff([rp(:,1:3), rp(:,4:6) * 50])), 2)]);
        fprintf('  Mean FD: %.4f mm\n', fd);
        qc_lines{end+1} = sprintf('mean FD (mm)          : %.4f', fd); %#ok<SAGROW>

        % ---- assemble and write --------------------------------------
        R = [motion, compcor];
        R = R(:, std(R, 0, 1) > 0);                    % drop constant columns
        rank_R = rank(R);
        if rank_R < size(R, 2)
            warning('Nuisance matrix is rank deficient (%d of %d columns).', ...
                    rank_R, size(R, 2));
        end

        mr_path = fullfile(denoise_dir, 'multiple_regressors.txt');
        save(mr_path, 'R', '-ascii', '-double');
        fprintf('  Nuisance regressors: %d columns (rank %d) -> %s\n', ...
                size(R, 2), rank_R, mr_path);
        qc_lines{end+1} = sprintf('total regressors      : %d (rank %d)', ...
                                  size(R, 2), rank_R); %#ok<SAGROW>

        fid = fopen(fullfile(denoise_dir, 'nuisance_qc.txt'), 'w');
        fprintf(fid, '%s\n', qc_lines{:});
        fclose(fid);

        % ---------------------------------------------------------
        % Job 1: fMRI model specification
        % ---------------------------------------------------------
        clear matlabbatch;
        scans = cell(nT, 1);
        for v = 1:nT
            scans{v} = sprintf('%s,%d', func_file, v);
        end

        matlabbatch{1}.spm.stats.fmri_spec.dir            = {model_dir};
        matlabbatch{1}.spm.stats.fmri_spec.timing.units   = 'secs';
        matlabbatch{1}.spm.stats.fmri_spec.timing.RT      = TR;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t  = N_SLICES;
        matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = ONSET_SLICE;
        matlabbatch{1}.spm.stats.fmri_spec.sess.scans     = scans;
        matlabbatch{1}.spm.stats.fmri_spec.sess.cond      = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi     = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess.regress   = struct('name', {}, 'val', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {mr_path};
        matlabbatch{1}.spm.stats.fmri_spec.sess.hpf       = HPF;
        matlabbatch{1}.spm.stats.fmri_spec.fact           = struct('name', {}, 'levels', {});
        matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
        matlabbatch{1}.spm.stats.fmri_spec.volt           = 1;
        matlabbatch{1}.spm.stats.fmri_spec.global         = 'None';
        matlabbatch{1}.spm.stats.fmri_spec.mthresh        = MTHRESH;
        matlabbatch{1}.spm.stats.fmri_spec.mask           = {''};
        matlabbatch{1}.spm.stats.fmri_spec.cvi            = 'AR(1)';

        % ---------------------------------------------------------
        % Job 2: model estimation, write residuals
        % ---------------------------------------------------------
        matlabbatch{2}.spm.stats.fmri_est.spmmat           = {fullfile(model_dir, 'SPM.mat')};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals  = 1;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

        spm_jobman('run', matlabbatch);

        nres = numel(dir(fullfile(model_dir, 'Res_*.nii')));
        fprintf('  Residual volumes written: %d (expected %d)\n', nres, nT);
        if nres ~= nT
            warning('Residual count does not match volume count for %s', subj);
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

%% ================= LOCAL FUNCTIONS =================

function roi = reslice_prob_map(tissue_path, ref_vol, dims)
% Resample a tissue probability map onto the functional grid.
% Trilinear interpolation, then the caller thresholds.  A distinct prefix
% avoids colliding with the rc1 file gm_no4 writes.
    flags = struct('interp', 1, 'mask', 1, 'mean', 0, 'which', 1, ...
                   'wrap', [0 0 0]', 'prefix', 'noiseroi_');
    spm_reslice({ref_vol.fname, tissue_path}, flags);
    [p, n, e] = fileparts(tissue_path);
    rp_path   = fullfile(p, ['noiseroi_' n e]);
    roi = spm_read_vols(spm_vol(rp_path));
    roi(~isfinite(roi)) = 0;
    if ~isequal(size(roi), dims(1:3))
        error('Resliced tissue map is %s, functional is %s.', ...
              mat2str(size(roi)), mat2str(dims(1:3)));
    end
end

function M = imerode_26(M)
% One iteration of 26-connected binary erosion, without Image Processing
% Toolbox.  A voxel survives only if all 26 neighbours are also set.
    keep = M;
    for di = -1:1
        for dj = -1:1
            for dk = -1:1
                if di == 0 && dj == 0 && dk == 0, continue; end
                keep = keep & circshift(M, [di dj dk]);
            end
        end
    end
    % circshift wraps; clear the outer shell so wrapped voxels cannot survive
    keep([1 end], :, :) = false;
    keep(:, [1 end], :) = false;
    keep(:, :, [1 end]) = false;
    M = keep;
end

function [comp, varexp] = compcor_components(X, k)
% Principal components of noise-ROI time series (Behzadi et al. 2007).
%   X : [nT x nVox] time series inside the ROI
%   k : number of components
% Each voxel has its mean and linear trend removed and is then variance
% normalised, so no single high-variance voxel dominates the decomposition.
    X = double(X);
    X(:, ~all(isfinite(X), 1)) = [];
    X = detrend(X);                         % removes mean and linear trend
    sd = std(X, 0, 1);
    X  = X(:, sd > 0) ./ sd(sd > 0);
    if isempty(X)
        comp = []; varexp = []; return;
    end
    k = min(k, min(size(X)));
    [U, S, ~] = svd(X, 'econ');
    ev = diag(S).^2;
    comp   = U(:, 1:k);
    varexp = ev(1:k) / sum(ev);
end

function Rm = expand_motion(rp, order)
% Realignment parameters expanded to the requested order.
%   6  : parameters
%   12 : parameters + temporal derivatives
%   24 : Friston-24 (parameters, derivatives, and the squares of both)
    d = [zeros(1, size(rp, 2)); diff(rp)];
    switch order
        case 6
            Rm = rp;
        case 12
            Rm = [rp, d];
        case 24
            Rm = [rp, d, rp.^2, d.^2];
        otherwise
            error('MOVEMENT_ORDER must be 6, 12, or 24 (got %d).', order);
    end
end
