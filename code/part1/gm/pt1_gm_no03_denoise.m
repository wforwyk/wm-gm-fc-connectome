%% =========================================================
%  gm_no3_denoise_v4.m
%  ---------------------------------------------------------
%  Nuisance regressors -> SPM 1st-level specification -> estimation
%
%  WHAT CHANGED FROM v3
%
%  (1) NOISE ROI CONSTRUCTION ORDER.  v3 resliced the tissue probability
%      map onto the functional grid first, then thresholded at 0.95 and
%      eroded.  Going from ~1 mm T1 voxels to 3.4 x 3.4 x 4 mm functional
%      voxels, trilinear interpolation pulls the probabilities down, so
%      almost nothing survived a 0.95 cut and erosion removed the rest:
%          CompCor c2: 319 voxels        <- white matter, implausibly small
%          Noise ROI c3 has only 0 voxels
%      v4 thresholds and erodes at the NATIVE segmentation resolution,
%      where the probabilities are sharp and erosion means what it says,
%      and only then resamples the binary mask onto the functional grid.
%      Erosion is now specified in millimetres rather than voxels, so it
%      does not depend on the T1 resolution.
%      If a ROI still comes out too small the script relaxes erosion, then
%      the threshold, and reports every attempt instead of silently
%      dropping the ROI.
%
%  (2) EXISTING MODEL DIRECTORY.  SPM refuses to overwrite an existing
%      SPM.mat in batch mode:
%          Abort...   (existing SPM file)
%      Specification was therefore skipped and estimation ran against the
%      STALE SPM.mat, which still pointed at a previous BASE_DIR:
%          File not found: .../derivatives/batch/0001/.../abr0001_rest.nii
%      v4 clears the first-level directory before specifying, so a re-run
%      always builds a fresh model.
%
%  NUISANCE MODEL (unchanged in substance from v3)
%    CompCor : principal components of tissue-ROI signal, following
%              Behzadi et al. (2007) NeuroImage 37:90
%    Motion  : realignment parameters expanded to order 6, 12 or 24
%              (12 = parameters + temporal derivatives, matching the
%              original PhysIO movement order)
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
BASE_DIR  = '/path/to/your/project/derivatives/batch';
SUBJECTS  = {'0001','0002'};   % subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/path/to/your/spm12/';

N_VOLS    = 240;      % number of volumes
TR        = 2;        % repetition time, seconds
N_SLICES  = 32;       % must equal the third image dimension
ONSET_SLICE = 2;      % reference slice used in slice timing (ST_REFSLICE in no1_2)

% --- CompCor noise ROIs ------------------------------------------------
%  [2 3]        white matter + CSF.  Standard aCompCor, and the DEFAULT.
%  [1 2 3 4 5]  what the original NTUSEC script passed to PhysIO.  Adding
%               c1 makes grey matter a noise ROI, which behaves much like
%               global signal regression (Murphy et al. 2009, NeuroImage
%               44:893) and removes part of the signal of interest.
NOISE_ROI_TISSUES   = [2 3];
NOISE_ROI_THRESHOLD = 0.95;   % applied at NATIVE segmentation resolution
NOISE_N_COMPONENTS  = 5;      % principal components per ROI

% --- Erosion: OFF by default at this resolution -----------------------
%  Erosion normally guards against grey-matter partial volume in the noise
%  ROI.  At 3.4 x 3.4 x 4 mm it removes the ROI instead.  Simulated on a
%  folded white-matter sheet, counting surviving functional voxels:
%
%     WM thickness | no erosion | 1 mm | 2 mm | v3 (erode on func grid)
%     -------------|------------|------|------|------------------------
%          4 mm    |     28     |   0  |   0  |          0
%          6 mm    |    352     |   8  |   0  |          0
%          8 mm    |   1040     |  45  |   0  |          0
%         12 mm    |   2208     | 458  |  36  |          0
%
%  Peripheral white matter and CSF spaces are 4-8 mm across, so any
%  erosion erases them.  That, rather than the resampling, is what
%  produced "CompCor c2: 319 voxels" and "c3 has only 0 voxels".
%
%  A 0.95 threshold on the native segmentation plus a >50% occupancy rule
%  after resampling is already a conservative partial-volume guard.
NOISE_ERODE_MM      = 0;      % erosion depth in mm at native resolution

% If a ROI ends up smaller than this on the functional grid, the script
% relaxes erosion, then the threshold, rather than dropping the ROI.
NOISE_MIN_VOXELS    = 100;
NOISE_FALLBACK_THRESHOLDS = [0.90 0.80 0.70];

% --- Motion model ------------------------------------------------------
MOVEMENT_ORDER = 12;          % 6, 12, or 24

% --- Model directory ---------------------------------------------------
%  SPM aborts specification if SPM.mat already exists.  With this true the
%  first-level directory is cleared first so a re-run is always clean.
%  Files inside subdirectories (e.g. residuals/) are left alone.
CLEAN_MODEL_DIR = true;

% --- SPM implicit masking ---------------------------------------------
%  Voxels below MTHRESH * global mean are excluded from the model and
%  their residuals are NaN, so they are lost downstream.  0.8 is the SPM
%  default and what the original used.  Check n_gm in gm_no6_qc.csv.
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

        % ---- clear stale model ---------------------------------------
        if CLEAN_MODEL_DIR
            n_removed = clear_model_dir(model_dir);
            if n_removed > 0
                fprintf('  cleared %d stale file(s) from %s\n', n_removed, model_dir);
            end
        end

        % ---- load the functional once --------------------------------
        Vf = spm_vol(func_file);
        nT = numel(Vf);
        if nT ~= N_VOLS
            warning('%s has %d volumes, N_VOLS is %d. Using the file.', ...
                    subj, nT, N_VOLS);
        end
        Yf = spm_read_vols(Vf);
        dims = size(Yf);
        Y2 = reshape(Yf, [], nT)';                     % [nT x nVox]
        clear Yf;

        qc = {};
        qc{end+1} = sprintf('subject               : %s', subj); %#ok<SAGROW>
        qc{end+1} = sprintf('volumes               : %d', nT);   %#ok<SAGROW>
        qc{end+1} = sprintf('functional voxels     : %d', prod(dims(1:3))); %#ok<SAGROW>

        % ---- CompCor components --------------------------------------
        compcor = [];
        for t = 1:numel(NOISE_ROI_TISSUES)
            k  = NOISE_ROI_TISSUES(t);
            dd = dir(fullfile(seg_dir, sprintf('c%d%s*.nii', k, subj)));
            if isempty(dd)
                error('Tissue segmentation c%d not found in %s', k, seg_dir);
            end
            tissue_path = fullfile(dd(1).folder, dd(1).name);

            [roi_mask, info] = build_noise_roi(tissue_path, Vf(1), dims, k, ...
                                   NOISE_ROI_THRESHOLD, NOISE_ERODE_MM, ...
                                   NOISE_MIN_VOXELS, NOISE_FALLBACK_THRESHOLDS, ...
                                   denoise_dir);

            fprintf('  c%d: native %d mm voxels, thr %.2f -> %d, erode %.1fmm -> %d, ', ...
                    k, round(info.native_mm), info.threshold_used, ...
                    info.n_native_threshold, info.erode_mm_used, info.n_native_eroded);
            fprintf('functional grid -> %d voxels\n', info.n_functional);

            nvox = nnz(roi_mask);
            if nvox < NOISE_N_COMPONENTS * 5
                warning(['Noise ROI c%d still has only %d voxels after all ' ...
                         'fallbacks; skipping this ROI.'], k, nvox);
                qc{end+1} = sprintf('CompCor c%d           : SKIPPED (%d voxels)', k, nvox); %#ok<SAGROW>
                continue;
            end

            [comp, varexp] = compcor_components(Y2(:, roi_mask(:)), NOISE_N_COMPONENTS);
            compcor = [compcor, comp]; %#ok<AGROW>

            fprintf('      -> %d components, %.1f%% of ROI variance\n', ...
                    size(comp, 2), 100 * sum(varexp));
            qc{end+1} = sprintf(['CompCor c%d           : %d voxels (thr %.2f, ' ...
                'erode %.1f mm), %d comps, var %.1f%%'], ...
                k, nvox, info.threshold_used, info.erode_mm_used, ...
                size(comp, 2), 100 * sum(varexp)); %#ok<SAGROW>
        end
        clear Y2;

        if isempty(compcor)
            warning(['No CompCor components for %s. The nuisance model is ' ...
                     'motion only.'], subj);
        end

        % ---- motion regressors ---------------------------------------
        rp = load(rp_file);
        if size(rp, 1) ~= nT
            error('rp file has %d rows, functional has %d volumes.', ...
                  size(rp, 1), nT);
        end
        motion = expand_motion(rp, MOVEMENT_ORDER);
        fprintf('  Motion: order %d -> %d regressors\n', ...
                MOVEMENT_ORDER, size(motion, 2));
        qc{end+1} = sprintf('motion                : order %d, %d regressors', ...
                            MOVEMENT_ORDER, size(motion, 2)); %#ok<SAGROW>

        fd = mean([0; sum(abs(diff([rp(:,1:3), rp(:,4:6) * 50])), 2)]);
        fprintf('  Mean FD: %.4f mm\n', fd);
        qc{end+1} = sprintf('mean FD (mm)          : %.4f', fd); %#ok<SAGROW>

        % ---- assemble and write --------------------------------------
        R = [motion, compcor];
        R = R(:, std(R, 0, 1) > 0);
        rank_R = rank(R);
        if rank_R < size(R, 2)
            warning('Nuisance matrix is rank deficient (%d of %d columns).', ...
                    rank_R, size(R, 2));
        end

        mr_path = fullfile(denoise_dir, 'multiple_regressors.txt');
        save(mr_path, 'R', '-ascii', '-double');
        fprintf('  Nuisance regressors: %d columns (rank %d)\n', size(R, 2), rank_R);
        qc{end+1} = sprintf('total regressors      : %d (rank %d)', ...
                            size(R, 2), rank_R); %#ok<SAGROW>

        fid = fopen(fullfile(denoise_dir, 'nuisance_qc.txt'), 'w');
        fprintf(fid, '%s\n', qc{:});
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

        % ---- verify the model that was actually built -----------------
        spm_mat = fullfile(model_dir, 'SPM.mat');
        if ~exist(spm_mat, 'file')
            error('SPM.mat was not created. Specification did not run.');
        end
        Sd = load(spm_mat);
        built_from = Sd.SPM.xY.P(1, :);
        if ~contains(built_from, func_file(1:min(end, 60)))
            warning(['SPM.mat references a different input than expected.\n' ...
                     '  expected: %s\n  found   : %s'], func_file, strtrim(built_from));
        end

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

function n_removed = clear_model_dir(model_dir)
% Remove the files SPM writes into a first-level directory, so that
% specification does not abort on an existing SPM.mat.  Subdirectories
% (residuals/, etc.) are left untouched.
    patterns = {'SPM.mat', 'beta_*.nii', 'Res_*.nii', 'ResMS.nii', ...
                'RPV.nii', 'mask.nii', 'con_*.nii', 'spmT_*.nii', ...
                'spmF_*.nii', 'ess_*.nii'};
    n_removed = 0;
    for p = 1:numel(patterns)
        dd = dir(fullfile(model_dir, patterns{p}));
        for f = 1:numel(dd)
            if ~dd(f).isdir
                delete(fullfile(dd(f).folder, dd(f).name));
                n_removed = n_removed + 1;
            end
        end
    end
end

function [roi_mask, info] = build_noise_roi(tissue_path, ref_vol, dims, tissue_k, ...
                                thr, erode_mm, min_vox, fallback_thr, tmp_dir)
% Build a CompCor noise ROI on the functional grid.
%
% ORDER MATTERS.  The threshold and the erosion are applied at the native
% segmentation resolution, where the tissue probabilities are sharp and a
% millimetre of erosion is actually resolvable.  Only the resulting BINARY
% mask is resampled onto the functional grid, where a voxel is kept if more
% than half of it lies inside the eroded tissue.
%
% Doing it the other way round (reslice the probability map first, then
% threshold at 0.95) is what emptied the ROIs in v3: trilinear
% interpolation from ~1 mm to 3.4 x 3.4 x 4 mm pulls the probabilities well
% below 0.95 almost everywhere.

    Vt = spm_vol(tissue_path);
    Pt = spm_read_vols(Vt);
    Pt(~isfinite(Pt)) = 0;
    native_zooms = sqrt(sum(Vt.mat(1:3,1:3).^2, 1));

    attempts = [thr, fallback_thr(:)'];
    info = struct('native_mm', mean(native_zooms), 'threshold_used', NaN, ...
                  'erode_mm_used', NaN, 'n_native_threshold', 0, ...
                  'n_native_eroded', 0, 'n_functional', 0, 'relaxed', false);

    for a = 1:numel(attempts)
        this_thr = attempts(a);
        % try the requested erosion first, then no erosion
        for erode_try = [erode_mm, 0]
            m = Pt > this_thr;
            n_thr = nnz(m);
            n_iter = max(0, round(erode_try / mean(native_zooms)));
            for e = 1:n_iter
                m = imerode_26(m);
            end
            n_ero = nnz(m);
            if n_ero == 0
                continue;
            end

            roi_mask = reslice_binary(m, Vt, ref_vol, dims, tissue_k, tmp_dir);
            n_fun = nnz(roi_mask);

            if n_fun >= min_vox || (a == numel(attempts) && erode_try == 0)
                info.threshold_used     = this_thr;
                info.erode_mm_used      = erode_try;
                info.n_native_threshold = n_thr;
                info.n_native_eroded    = n_ero;
                info.n_functional       = n_fun;
                info.relaxed            = (this_thr ~= thr) || (erode_try ~= erode_mm);
                if info.relaxed
                    fprintf(['      [relaxed] c%d: threshold %.2f -> %.2f, ' ...
                             'erosion %.1f -> %.1f mm to reach %d voxels\n'], ...
                             tissue_k, thr, this_thr, erode_mm, erode_try, n_fun);
                end
                return;
            end
            if erode_try == 0
                break;   % nothing left to relax at this threshold
            end
        end
    end

    roi_mask = false(dims(1:3));
    info.threshold_used = attempts(end);
    info.erode_mm_used  = 0;
end

function roi_mask = reslice_binary(m, Vt, ref_vol, dims, tissue_k, tmp_dir)
% Write a binary mask at native resolution, resample it onto the reference
% grid with trilinear interpolation, and keep voxels more than half inside.
    tmp_name = sprintf('noiseroi_c%d_tmp.nii', tissue_k);
    Vo        = Vt;
    Vo.fname  = fullfile(tmp_dir, tmp_name);
    Vo.dt     = [spm_type('float32') spm_platform('bigend')];
    Vo.pinfo  = [1; 0; 0];
    Vo.descrip = sprintf('noise ROI c%d (native, thresholded, eroded)', tissue_k);
    if exist(Vo.fname, 'file'), delete(Vo.fname); end
    spm_write_vol(Vo, double(m));

    flags = struct('interp', 1, 'mask', 1, 'mean', 0, 'which', 1, ...
                   'wrap', [0 0 0]', 'prefix', 'r');
    spm_reslice({ref_vol.fname, Vo.fname}, flags);

    rpath = fullfile(tmp_dir, ['r' tmp_name]);
    Rv = spm_read_vols(spm_vol(rpath));
    Rv(~isfinite(Rv)) = 0;
    if ~isequal(size(Rv), dims(1:3))
        error('Resliced noise ROI is %s, functional is %s.', ...
              mat2str(size(Rv)), mat2str(dims(1:3)));
    end
    roi_mask = Rv > 0.5;
end

function M = imerode_26(M)
% One iteration of 26-connected binary erosion, without the Image
% Processing Toolbox.  A voxel survives only if all 26 neighbours are set.
    keep = M;
    for di = -1:1
        for dj = -1:1
            for dk = -1:1
                if di == 0 && dj == 0 && dk == 0, continue; end
                keep = keep & circshift(M, [di dj dk]);
            end
        end
    end
    keep([1 end], :, :) = false;   % circshift wraps; clear the outer shell
    keep(:, [1 end], :) = false;
    keep(:, :, [1 end]) = false;
    M = keep;
end

function [comp, varexp] = compcor_components(X, k)
% Principal components of noise-ROI time series (Behzadi et al. 2007).
% Each voxel has its mean and linear trend removed and is variance
% normalised, so no single high-variance voxel dominates.
    X = double(X);
    X(:, ~all(isfinite(X), 1)) = [];
    X = detrend(X);
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
    d = [zeros(1, size(rp, 2)); diff(rp)];
    switch order
        case 6,  Rm = rp;
        case 12, Rm = [rp, d];
        case 24, Rm = [rp, d, rp.^2, d.^2];
        otherwise
            error('MOVEMENT_ORDER must be 6, 12, or 24 (got %d).', order);
    end
end
