% ============================================================
% pt1_template_no05_coregister_endpoints_to_fmri.m
% ------------------------------------------------------------
% PURPOSE : Reslice each TRACULA endpoint path-density (PD) image from
%           DWI space onto the subject's fMRI grid. These images provide
%           the voxel-level WM-endpoint labels required by Part 2 Aim 2.
%
% PIPELINE CONTEXT:
%   TRACULA endpt{1,2}.pd.nii.gz (DWI space; continuous PD image)
%       ↓ SPM Coregister: Reslice; ref = bnbmdenoised_detrended_rest_{subject}.nii
%   fmri_endpt{1,2}.pd.nii (exact BnB/R/W analysis grid)
%
% IMPORTANT:
%   - This is RESLICING ONLY. It does not estimate a DWI-to-fMRI
%     registration. Run it only after visual QC has established that the
%     DWI and fMRI references are correctly aligned in subject space.
%   - Trilinear interpolation (interp = 1) is intentional: endpoint PD is
%     continuous, unlike the integer-valued AAL labels in template no03.
%   - The original .nii.gz endpoint maps are retained. Their uncompressed
%     .nii copies are also retained because SPM reslices uncompressed NIfTI.
%
% INPUT (per subject):
%   gm/func/derived/bnbmdenoised_detrended_rest_{subject}.nii
%   wm/freesurfer/{subject}/dpath/{tract}_avg16_syn_bbr/endpt{1,2}.pd.nii.gz
%
% OUTPUT (per tract):
%   wm/freesurfer/{subject}/dpath/{tract}_avg16_syn_bbr/fmri_endpt{1,2}.pd.nii
%
% NEXT STEP:
%   Run pt1_template_no04_veritifaction.sh with VERIFY_ENDPOINTS=true,
%   then run part2/pt2_no04_endpoint_voxel_label.py.
% ============================================================

%% ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = {'0001','0002'};  % Numeric subject IDs.
BASE_DIR = '/path/to/your/project/derivatives/batch';

% TRACULA tract-directory suffix. Do not change unless TRACULA outputs differ.
TRACT_DIR_SUFFIX = '_avg16_syn_bbr';
% ─────────────────────────────────────────────────────────────

spm('defaults', 'fmri');
spm_jobman('initcfg');

failed = {};
processed = 0;

for i = 1:numel(SUBJECTS)
    subj = SUBJECTS{i};
    func_ref = fullfile(BASE_DIR, subj, 'gm', 'func', 'derived', ...
                        ['bnbmdenoised_detrended_rest_' subj '.nii']);
    dpath_dir = fullfile(BASE_DIR, subj, 'wm', 'freesurfer', subj, 'dpath');

    fprintf('\n[%s]\n', subj);
    if ~exist(func_ref, 'file') || ~exist(dpath_dir, 'dir')
        fprintf('  ERROR: missing BnB fMRI reference or TRACULA dpath directory\n');
        failed{end+1} = subj;
        continue
    end

    tract_dirs = dir(fullfile(dpath_dir, ['*' TRACT_DIR_SUFFIX]));
    tract_dirs = tract_dirs([tract_dirs.isdir]);
    if isempty(tract_dirs)
        fprintf('  ERROR: no tract directories ending %s\n', TRACT_DIR_SUFFIX);
        failed{end+1} = subj;
        continue
    end

    subject_failed = false;
    for t = 1:numel(tract_dirs)
        tract_dir = fullfile(dpath_dir, tract_dirs(t).name);
        for endpoint = 1:2
            gz_file = fullfile(tract_dir, sprintf('endpt%d.pd.nii.gz', endpoint));
            nii_file = fullfile(tract_dir, sprintf('endpt%d.pd.nii', endpoint));
            out_file = fullfile(tract_dir, sprintf('fmri_endpt%d.pd.nii', endpoint));

            if exist(out_file, 'file')
                fprintf('  %s endpt%d: output exists; skip\n', tract_dirs(t).name, endpoint);
                continue
            end
            if ~exist(gz_file, 'file') && ~exist(nii_file, 'file')
                fprintf('  ERROR: %s endpt%d source missing\n', tract_dirs(t).name, endpoint);
                subject_failed = true;
                continue
            end
            if ~exist(nii_file, 'file')
                try
                    gunzip(gz_file, tract_dir);
                catch ME
                    fprintf('  ERROR: cannot extract %s (%s)\n', gz_file, ME.message);
                    subject_failed = true;
                    continue
                end
            end

            try
                matlabbatch = {};
                matlabbatch{1}.spm.spatial.coreg.write.ref = {[func_ref ',1']};
                matlabbatch{1}.spm.spatial.coreg.write.source = {[nii_file ',1']};
                matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
                matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
                matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
                matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'fmri_';
                spm_jobman('run', matlabbatch);
                fprintf('  %s endpt%d: OK\n', tract_dirs(t).name, endpoint);
                processed = processed + 1;
            catch ME
                fprintf('  ERROR: %s endpt%d (%s)\n', tract_dirs(t).name, endpoint, ME.message);
                subject_failed = true;
            end
        end
    end
    if subject_failed
        failed{end+1} = subj;
    end
end

fprintf('\n============================================\n');
fprintf('  Resliced endpoint maps: %d\n', processed);
if isempty(failed)
    fprintf('  All subjects completed without errors.\n');
else
    fprintf('  SUBJECTS WITH ERRORS (%d):\n', numel(failed));
    fprintf('    %s\n', strjoin(failed, ', '));
end
fprintf('============================================\n');
