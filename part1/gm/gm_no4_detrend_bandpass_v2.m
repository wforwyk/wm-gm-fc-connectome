%% =========================================================
%  gm_no4_detrend_bandpass_v2.m
%  ---------------------------------------------------------
%  Residuals -> linear detrend -> band-pass filter -> single 4-D file
%
%  RESTORED FROM THE ORIGINAL PIPELINE
%    denoise_script_detrend_masking_part.m
%
%  The v1 rewrite (gm_no4_modelestimation_detrend.m) performed the linear
%  detrend but dropped the band-pass step
%      rp_bandpass(detrend_dir, 2, 0.08, 0.009, 'Yes', 0)
%  Resting-state FC is conventionally computed in the 0.01-0.08 Hz band.
%  Without the low-pass side, aliased cardiac and respiratory components
%  enter every correlation.  The GLM high-pass filter (128 s ~ 0.008 Hz)
%  covers the low-frequency side only.
%
%  The filter here is implemented directly, in the same FFT form used by
%  the REST toolbox: zero-pad to a power of two, null the frequency bins
%  outside the pass band, inverse transform, truncate.  No Signal
%  Processing Toolbox required, and no dependency on rp_bandpass.
%
%  Model estimation now lives in gm_no3_denoise_v2.m, so this script does
%  post-processing only.
%
%  INPUT   {BASE_DIR}/{subj}/gm/func/model/1st_level/Res_*.nii
%          {BASE_DIR}/{subj}/gm/anat/segmentation/c1{subj}*.nii
%  OUTPUT  {BASE_DIR}/{subj}/gm/func/derived/denoised_detrended_rest_{subj}.nii
%          {BASE_DIR}/{subj}/gm/func/model/1st_level/residuals/  (Res_* moved)
%
%  NEXT    gm_no5_masking_binarise_v2.m
%% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch';
SUBJECTS  = {'1123','1170'};   % subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/opt/spm/';

TR            = 2;         % repetition time, seconds
BAND_LOW_HZ   = 0.009;     % high-pass edge  (original rp_bandpass value)
BAND_HIGH_HZ  = 0.08;      % low-pass  edge  (original rp_bandpass value)
DO_BANDPASS   = true;      % set false to reproduce the v1 behaviour

% Voxels included in the detrend / filter operation.  This is a permissive
% working mask only; the analysis mask is set in gm_no5.
C1_THRESHOLD  = 0.2;
%% =================================================

addpath(SPM_ROOT);
spm('defaults', 'FMRI');
spm_jobman('initcfg');

failed_subjects = {};

for s = 1:numel(SUBJECTS)
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, numel(SUBJECTS), subj);

    seg_dir     = fullfile(BASE_DIR, subj, 'gm', 'anat', 'segmentation');
    model_dir   = fullfile(BASE_DIR, subj, 'gm', 'func', 'model', '1st_level');
    derived_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'derived');

    if ~exist(derived_dir, 'dir'), mkdir(derived_dir); end

    try
        % ---- residuals ------------------------------------------------
        res_files = dir(fullfile(model_dir, 'Res_*.nii'));
        if isempty(res_files)
            error('No Res_*.nii found in %s. Run gm_no3_denoise_v2 first.', model_dir);
        end
        res_paths = fullfile({res_files.folder}', {res_files.name}');

        V = spm_vol(char(res_paths));
        Y = spm_read_vols(V);
        sz = size(Y);
        nT = sz(4);
        fprintf('  residuals loaded: %dx%dx%d x %d volumes\n', sz(1), sz(2), sz(3), nT);

        % ---- working mask ---------------------------------------------
        c1_raw = pick_one(fullfile(seg_dir, ['c1' subj '*.nii']), 'c1 segmentation');
        resliced_mask_path = reslice_mask_to_data(c1_raw, V(1));
        Vm = spm_vol(resliced_mask_path);
        M  = spm_read_vols(Vm) > C1_THRESHOLD;

        % Residuals are NaN outside the SPM analysis mask; exclude those too.
        finite_all = ~any(isnan(Y), 4);
        M = M & finite_all;
        fprintf('  working mask: %d voxels (c1 > %.2f and finite residuals)\n', ...
                nnz(M), C1_THRESHOLD);
        if nnz(M) == 0
            error('Working mask is empty.');
        end

        % ---- detrend ---------------------------------------------------
        Y2       = reshape(Y, [], nT);
        mask_vec = M(:);
        Y2(mask_vec, :) = detrend(Y2(mask_vec, :)')';
        fprintf('  linear detrend done\n');

        % ---- band-pass -------------------------------------------------
        if DO_BANDPASS
            [Y2, kept, nyq] = fft_bandpass(Y2, mask_vec, TR, BAND_LOW_HZ, BAND_HIGH_HZ);
            fprintf('  band-pass %.3f-%.3f Hz applied (Nyquist %.3f Hz, %d bins retained)\n', ...
                    BAND_LOW_HZ, BAND_HIGH_HZ, nyq, kept);
        else
            fprintf('  band-pass SKIPPED (DO_BANDPASS = false)\n');
        end

        % Voxels outside the working mask carry no meaningful signal.
        % Setting them to NaN lets gm_no5 identify the valid set without
        % relying on the sign of any single time point.
        Y2(~mask_vec, :) = NaN;
        Yd = reshape(Y2, sz);

        % ---- write single 4-D file -------------------------------------
        output_name = fullfile(derived_dir, ['denoised_detrended_rest_' subj '.nii']);
        if exist(output_name, 'file'), delete(output_name); end

        Vout = V;
        for t = 1:numel(Vout)
            Vout(t).fname   = output_name;
            Vout(t).n       = [t 1];
            Vout(t).dt      = [spm_type('float32') spm_platform('bigend')];
            Vout(t).descrip = sprintf('detrended%s', ...
                              ternary(DO_BANDPASS, sprintf(' bandpass %.3f-%.3f Hz', ...
                              BAND_LOW_HZ, BAND_HIGH_HZ), ''));
            spm_write_vol(Vout(t), Yd(:,:,:,t));
        end
        fprintf('  wrote %s\n', output_name);

        % ---- archive residuals ------------------------------------------
        residual_dir = fullfile(model_dir, 'residuals');
        if ~exist(residual_dir, 'dir'), mkdir(residual_dir); end
        movefile(fullfile(model_dir, 'Res_*.nii'), residual_dir);

        fprintf('[OK] %s\n', subj);

    catch err
        fprintf('[FAILED] %s\n', subj);
        fprintf('  Error: %s\n', err.message);
        failed_subjects{end+1} = subj; %#ok<SAGROW>
    end
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

function [Y2, n_kept, nyquist] = fft_bandpass(Y2, mask_vec, TR, f_low, f_high)
% FFT band-pass in the form used by the REST toolbox.
%   Zero-pad to a power of two, null the bins outside [f_low, f_high] and
%   their negative-frequency mirrors, inverse transform, truncate.
%   Operates only on masked voxels; the rest are untouched.

    X   = Y2(mask_vec, :)';            % [nT x nVox]
    nT  = size(X, 1);
    nyquist = 1 / (2 * TR);

    if f_high >= nyquist
        error('BAND_HIGH_HZ (%.3f) must be below the Nyquist frequency (%.3f).', ...
              f_high, nyquist);
    end

    padded = 2^nextpow2(nT);
    Xp = [X; zeros(padded - nT, size(X, 2))];

    F = fft(Xp, [], 1);

    % Bin index of a frequency f is  f * padded * TR + 1.
    idxLow  = ceil(f_low  * padded * TR + 1);
    idxHigh = fix (f_high * padded * TR + 1);
    idxLow  = max(idxLow, 2);                       % never keep DC
    idxHigh = min(idxHigh, floor(padded / 2));      % never exceed Nyquist

    keep = false(padded, 1);
    keep(idxLow:idxHigh) = true;

    % Mirror onto the negative frequencies so the inverse transform is real.
    mirror = padded - (idxLow:idxHigh) + 2;
    mirror = mirror(mirror >= 2 & mirror <= padded);
    keep(mirror) = true;

    F(~keep, :) = 0;
    Xf = real(ifft(F, [], 1));

    Y2(mask_vec, :) = Xf(1:nT, :)';
    n_kept = idxHigh - idxLow + 1;
end

function fp = pick_one(glob_pattern, label)
    dd = dir(glob_pattern);
    if isempty(dd), error('Missing %s. Pattern: %s', label, glob_pattern); end
    [~, idx] = max([dd.datenum]);
    fp = fullfile(dd(idx).folder, dd(idx).name);
end

function resliced_path = reslice_mask_to_data(mask_path, target_vol)
    flags = struct('interp', 0, 'mask', 1, 'mean', 0, 'which', 1, ...
                   'wrap', [0 0 0]', 'prefix', 'r');
    spm_reslice({target_vol.fname, mask_path}, flags);
    [p, n, e]     = fileparts(mask_path);
    resliced_path = fullfile(p, ['r' n e]);
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
