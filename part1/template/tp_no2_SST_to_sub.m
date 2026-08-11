% ============================================================
% no2_SST_to_sub.m
% -----------------------------------------------------------
% PURPOSE : Warp sst_all_ROIs_AAL.nii from SST space into each
%           subject's native T1 space using the DARTEL flow field.
%           Direct .m conversion of SST_to_sub.mat.
%
% PIPELINE CONTEXT:
%   AAL (MNI space)
%       ↓ iy_NCKU_336Ss_6.nii  [inverse deformation field, MNI → SST]
%         → already applied; output = sst_all_ROIs_AAL.nii
%   sst_all_ROIs_AAL.nii  (SST space)   ← INPUT (shared, one file)
%       ↓ u_rc1sub-{subject}*.nii  [DARTEL flow field, SST → native T1]
%   wsst_all_ROIs_AAL_u_rc1sub-{subject}*.nii  (native T1 space)  ← OUTPUT
%
% INPUT   : {BASE_DIR}/{subject}/gm/derived/
%               u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii  (DARTEL flow field)
%           {TEMPLATE_DIR}/sst_all_ROIs_AAL.nii
%               AAL atlas in SST space (shared across all SUBJECTS)
%
% OUTPUT  : {BASE_DIR}/{subject}/gm/derived/
%               wsst_all_ROIs_AAL_u_rc1sub-{subject}_ses-1_T1w_NCKU_336Ss.nii
%
% NOTE    : SPM2025b must be on the MATLAB path.
%           SUBJECTS uses numeric IDs only (no sub- prefix).
%           Parameters match original SST_to_sub.mat exactly:
%             interp = 1 (trilinear), K = 6
% ============================================================

%% ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS  = {'1001','1002'};  # subject_id must be in ' ' and divided by ,

BASE_DIR  = '/your/base/directory/is/your/root/directory';

TEMPLATE_DIR = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/template/NCKU';
SST_ATLAS    = fullfile(TEMPLATE_DIR, 'sst_all_ROIs_AAL.nii');
% ─────────────────────────────────────────────────────────────

spm('defaults', 'fmri');
spm_jobman('initcfg');

if ~exist(SST_ATLAS, 'file')
    error('SST atlas not found: %s', SST_ATLAS);
end

failed = {};

for i = 1:numel(SUBJECTS)
    subj     = SUBJECTS{i};
    der_dir  = fullfile(BASE_DIR, subj, 'gm', 'derived');
    flowfile = fullfile(der_dir, ['u_rc1sub-' subj '_ses-1_T1w_NCKU_336Ss.nii']);

    fprintf('\n[%s]\n', subj);

    if ~exist(flowfile, 'file')
        fprintf('  ERROR: DARTEL flow field not found: %s\n', flowfile);
        failed{end+1} = subj;
        continue
    end

    fprintf('  flowfile : %s\n', flowfile);
    fprintf('  atlas    : %s\n', SST_ATLAS);

    % ── SPM batch: exact match to SST_to_sub.mat ────────────
    matlabbatch = {};
    matlabbatch{1}.spm.tools.dartel.crt_iwarped.flowfields = {flowfile};
    matlabbatch{1}.spm.tools.dartel.crt_iwarped.images     = {SST_ATLAS};
    matlabbatch{1}.spm.tools.dartel.crt_iwarped.K          = 6;
    matlabbatch{1}.spm.tools.dartel.crt_iwarped.interp     = 1;

    try
        spm_jobman('run', matlabbatch);
        fprintf('  OK: wsst_all_ROIs_AAL_u_rc1sub-%s_ses-1_T1w_NCKU_336Ss.nii\n', subj);
    catch ME
        fprintf('  ERROR: %s\n', ME.message);
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
