% ============================================================
% no3_coregister_to_dwi_fmri.m
% -----------------------------------------------------------
% PURPOSE : Coregister wsst_all_ROIs_AAL (native T1 space) into
%           (a) fMRI space → prefix: fmri_
%           (b) DWI space  → prefix: dwi_
%           using SPM coreg reslice (nearest neighbour, interp=0).
%
% PIPELINE CONTEXT:
%   wsst_all_ROIs_AAL_u_rc1sub-{subject}*.nii  (native T1, from step2)
%       ↓ coreg ref = mean{subject}_rest.nii
%       → fmri_wsst_all_ROIs_AAL_u_rc1sub-{subject}*.nii  (fMRI space)
%       ↓ coreg ref = dwi.nii (extracted from dwi.nii.gz)
%       → dwi_wsst_all_ROIs_AAL_u_rc1sub-{subject}*.nii   (DWI space)
%
% INPUT   : {BASE_DIR}/{subject}/gm/derived/
%               wsst_all_ROIs_AAL_u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
%           {BASE_DIR}/{subject}/gm/func/preprocessing/
%               mean{subject}_rest.nii
%           {BASE_DIR}/{subject}/wm/freesurfer{subject}/dmri/
%               dwi.nii.gz  → gunzip to dwi.nii before use
%
% OUTPUT  : {BASE_DIR}/{subject}/gm/derived/
%               fmri_wsst_all_ROIs_AAL_u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
%               dwi_wsst_all_ROIs_AAL_u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
%
% NOTE    : SPM2025b must be on the MATLAB path.
%           SUBJECTS uses numeric IDs only (no sub- prefix).
%           interp = 0 (nearest neighbour) — matches SST_coregister_to_dwi.mat,
%             preserves discrete AAL label integers.
%           interp in step2 (crt_iwarped) = 1: different SPM function,
%             continuous warp field requires trilinear.
%           dwi.nii.gz is extracted to dwi.nii and kept after processing.
%           prefix changed from original 'r' to 'fmri_'/'dwi_' for clarity.
% ============================================================

%% ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS  = {'1001','1002'};  # subject_id must be in ' ' and divided by ,

BASE_DIR  = '/your/base/directory/is/your/root/directory';

% ─────────────────────────────────────────────────────────────

spm('defaults', 'fmri');
spm_jobman('initcfg');

failed = {};

for i = 1:numel(SUBJECTS)
    subj = SUBJECTS{i};

    der_dir  = fullfile(BASE_DIR, subj, 'gm', 'derived');
    func_dir = fullfile(BASE_DIR, subj, 'gm', 'func', 'preprocessing');
    dwi_dir  = fullfile(BASE_DIR, subj, 'wm', 'freesurfer', subj, 'dmri');

    aal_t1   = fullfile(der_dir,  ['wsst_all_ROIs_AAL_u_rc1sub-' subj '_ses-1_T1w_NCKU_336Ss.nii']);
    fmri_ref = fullfile(func_dir, ['mean' subj '_rest.nii']);
    dwi_gz   = fullfile(dwi_dir,  'dwi.nii.gz');
    dwi_nii  = fullfile(dwi_dir,  'dwi.nii');

    fprintf('\n[%s]\n', subj);

    % ── check inputs ─────────────────────────────────────────
    missing = 0;
    for chk = {aal_t1, fmri_ref, dwi_gz}
        if ~exist(chk{1}, 'file')
            fprintf('  ERROR: not found: %s\n', chk{1});
            missing = 1;
        end
    end
    if missing
        failed{end+1} = subj;
        continue
    end

    % ── extract dwi.nii.gz → dwi.nii ─────────────────────────
    if ~exist(dwi_nii, 'file')
        fprintf('  Extracting dwi.nii.gz ... ');
        try
            gunzip(dwi_gz, dwi_dir);
            fprintf('OK\n');
        catch ME
            fprintf('ERROR: %s\n', ME.message);
            failed{end+1} = subj;
            continue
        end
    else
        fprintf('  dwi.nii already exists, skip extraction\n');
    end

    % ── (a) coreg AAL T1 → fMRI space ────────────────────────
    fprintf('  Coreg -> fMRI space ... ');
    try
        matlabbatch = {};
        matlabbatch{1}.spm.spatial.coreg.write.ref             = {[fmri_ref ',1']};
        matlabbatch{1}.spm.spatial.coreg.write.source          = {[aal_t1   ',1']};
        matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap   = [0 0 0];
        matlabbatch{1}.spm.spatial.coreg.write.roptions.mask   = 0;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'fmri_';
        spm_jobman('run', matlabbatch);
        fprintf('OK\n');
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        failed{end+1} = subj;
    end

    % ── (b) coreg AAL T1 → DWI space ─────────────────────────
    fprintf('  Coreg -> DWI space  ... ');
    try
        matlabbatch = {};
        matlabbatch{1}.spm.spatial.coreg.write.ref             = {[dwi_nii ',1']};
        matlabbatch{1}.spm.spatial.coreg.write.source          = {[aal_t1  ',1']};
        matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 0;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap   = [0 0 0];
        matlabbatch{1}.spm.spatial.coreg.write.roptions.mask   = 0;
        matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'dwi_';
        spm_jobman('run', matlabbatch);
        fprintf('OK\n');
    catch ME
        fprintf('ERROR: %s\n', ME.message);
        failed{end+1} = subj;
    end

end

fprintf('\n============================================\n');
if isempty(failed)
    fprintf('  All done. No failures.\n');
else
    fprintf('  FAILED SUBJECTS (%d):\n', numel(failed));
    for k = 1:numel(failed)
        fprintf('    %s\n', failed{k});
    end
end
fprintf('============================================\n');