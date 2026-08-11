%% =========================================================
%  no4_modelestimation_detrend.m
%  ---------------------------------------------------------
%  Step 1: Model Estimation
%  Step 2: Reslice Mask
%  Step 3: Detrend & Masking
%% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/your/base/directory/is/your/root/directory';
SUBJECTS  = {'1001','1002'};  # subject_id must be in ' ' and divided by ,

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

    seg_dir     = fullfile(BASE_DIR, subj, 'gm', 'anat', 'segmentation');
    model_dir   = fullfile(BASE_DIR, subj, 'gm', 'func', 'model', '1st_level');
    derived_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'derived');

    try
        % --- STEP 1: Model Estimation (output: Residuals) ---
        spm_mat_file = fullfile(model_dir, 'SPM.mat');

        if ~exist(spm_mat_file, 'file')
            error('Missing SPM.mat: %s', spm_mat_file);
        end

        fprintf('  [1/3] Estimating model...\n');
        clear matlabbatch;
        matlabbatch{1}.spm.stats.fmri_est.spmmat          = {spm_mat_file};
        matlabbatch{1}.spm.stats.fmri_est.write_residuals  = 1;
        matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
        spm_jobman('run', matlabbatch);

        % --- STEP 2: Reslice Mask ---
        fprintf('  [2/3] Reslicing c1 mask to match fMRI resolution...\n');

        res_sample  = dir(fullfile(model_dir, 'Res_0001.nii'));
        target_vol  = spm_vol(fullfile(res_sample.folder, res_sample.name));
        c1_raw      = pick_one(fullfile(seg_dir, 'c1*.nii'), 'c1 mask');
        resliced_mask_path = reslice_mask_to_data(c1_raw, target_vol);

        % --- STEP 3: Detrend & Masking ---
        fprintf('  [3/3] Detrending & Masking...\n');

        res_files = dir(fullfile(model_dir, 'Res_*.nii'));
        res_paths = fullfile({res_files.folder}', {res_files.name}');

        V  = spm_vol(char(res_paths));
        Y  = spm_read_vols(V);
        Vm = spm_vol(resliced_mask_path);
        M  = spm_read_vols(Vm) > 0.2;

        Yd = detrend_timeseries(Y, M);

        if ~exist(derived_dir, 'dir'), mkdir(derived_dir); end

        Vout = V;
        output_name = fullfile(derived_dir, ['denoised_detrended_rest_' subj '.nii']);
        for t = 1:numel(Vout)
            Vout(t).fname = output_name;
            Vout(t).n     = [t 1];
            spm_write_vol(Vout(t), Yd(:,:,:,t));
        end

        % --- STEP 4: Move Residuals ---
        fprintf('  [4/4] Moving Res_*.nii to residuals folder...\n');
        residual_dir = fullfile(model_dir, 'residuals');
        if ~exist(residual_dir, 'dir'), mkdir(residual_dir); end
        movefile(fullfile(model_dir, 'Res_*.nii'), residual_dir);

        fprintf('[OK] %s\n', subj);

    catch err
        fprintf('[FAILED] %s\n', subj);
        fprintf('  Error: %s\n', err.message);
        failed_subjects{end+1} = subj;
    end

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

%% ================= LOCAL FUNCTIONS =================

function Yd = detrend_timeseries(Y, mask)
    sz       = size(Y);
    Y2       = reshape(Y, [], sz(4));
    mask_vec = mask(:) > 0;
    Yd2      = Y2;
    Yd2(mask_vec, :) = detrend(Y2(mask_vec, :)')';
    Yd = reshape(Yd2, sz);
end

function fp = pick_one(glob_pattern, label)
    dd = dir(glob_pattern);
    if isempty(dd), error('Missing %s. Pattern: %s', label, glob_pattern); end
    [~, idx] = max([dd.datenum]);
    fp = fullfile(dd(idx).folder, dd(idx).name);
end

function resliced_path = reslice_mask_to_data(mask_path, target_vol)
    flags = struct('interp', 0, 'mask', 1, 'mean', 0, 'which', 1, 'wrap', [0 0 0]', 'prefix', 'r');
    spm_reslice({target_vol.fname, mask_path}, flags);
    [p, n, e]     = fileparts(mask_path);
    resliced_path = fullfile(p, ['r' n e]);
end
