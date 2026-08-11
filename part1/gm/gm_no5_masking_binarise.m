%% =========================================================
%  no5_masking_binarise.m
%  ---------------------------------------------------------
%  Step 1: Masking (spm_mask, prefix 'm')
%  Step 2: Binarisation (prefix 'bm')
%% =========================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/your/base/directory/is/your/root/directory';
SUBJECTS  = {'1001','1002'};  # subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/opt/spm/';

%% ================= SPM INITIALIZATION =================
addpath(SPM_ROOT);
spm('defaults', 'FMRI');

%% ================= MAIN LOOP =================
failed_subjects = {};

for s = 1:length(SUBJECTS)
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, length(SUBJECTS), subj);

    seg_dir     = fullfile(BASE_DIR, subj, 'gm', 'anat', 'segmentation');
    derived_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'derived');

    try
        % --- STEP 1: Masking (output: m* file) ---
        fprintf('  [1/2] Masking (threshold 0.9)...\n');

        c1_file   = fullfile(seg_dir, ['c1' subj '_t1.nii']);
        func_file = fullfile(derived_dir, ['denoised_detrended_rest_' subj '.nii']);

        if ~exist(c1_file, 'file'),   error('Missing c1 file: %s',   c1_file);   end
        if ~exist(func_file, 'file'), error('Missing func file: %s', func_file); end

        spm_mask(c1_file, func_file, 0.9);

        % --- STEP 2: Binarisation (output: bm* file) ---
        fprintf('  [2/2] Binarising masked file...\n');

        masked_file = fullfile(derived_dir, ['mdenoised_detrended_rest_' subj '.nii']);

        if ~exist(masked_file, 'file')
            error('Missing masked file: %s', masked_file);
        end

        V_all = spm_vol(masked_file);
        img   = spm_read_vols(V_all);

        [fpath, fname, fext] = fileparts(masked_file);
        Vout         = V_all(1);
        Vout.fname   = fullfile(fpath, ['b' fname fext]);
        Vout.descrip = 'Binarised Mask';
        spm_write_vol(Vout, +(img(:,:,:,1) > 0));

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
