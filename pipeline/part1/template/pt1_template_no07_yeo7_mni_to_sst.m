% ============================================================
% pt1_template_no07_yeo7_mni_to_sst.m
% ------------------------------------------------------------
% PURPOSE
%   Put the categorical Yeo 2011 7-network MNI152 volume into
%   the project SST space. Run this atlas-level operation once;
%   it is not participant-level preprocessing.
%
% WHY THIS EXISTS
%   Aim 2 source and target identities are AAL3 ROI IDs. The Yeo
%   map must share the AAL3 MNI -> SST lineage before voxel overlap
%   can assign one canonical network to each cortical AAL3 ROI.
%   Do not push this atlas through individual DARTEL or fMRI reslices.
%
% INPUT
%   - Official Yeo2011 7-network LiberalMask MNI152 volume in YEO_DIR
%   - MNI -> SST inverse deformation field
%
% OUTPUT
%   {YEO_DIR}/sst/sst_Yeo2011_7Networks_MNI152_
%   FreeSurferConformed1mm_LiberalMask.nii
%
% IMPORTANT
%   Yeo values are categorical labels 0..7. Use nearest-neighbour
%   interpolation (interp = 0), never trilinear interpolation.
% ============================================================

%% USER SETTINGS ------------------------------------------------
YEO_DIR = '/path/to/your/project/derivatives/template/Yeo_JNeurophysiol11_MNI152';
SST_TEMPLATE_DIR = '/path/to/your/project/derivatives/template/sst_atlas';
DEFORMATION_FILE = fullfile(SST_TEMPLATE_DIR, 'iy_YOUR_DARTEL_TEMPLATE_6.nii');
OUTPUT_DIR = fullfile(YEO_DIR, 'sst');
% ---------------------------------------------------------------

spm('defaults', 'fmri');
spm_jobman('initcfg');

if ~exist(DEFORMATION_FILE, 'file')
    error('MNI -> SST deformation is missing: %s', DEFORMATION_FILE);
end
if ~exist(YEO_DIR, 'dir')
    error('Yeo directory is missing: %s', YEO_DIR);
end
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

yeo_file = resolve_yeo7_liberal_mask(YEO_DIR);
fprintf('Yeo MNI label map: %s\n', yeo_file);
fprintf('MNI -> SST field : %s\n', DEFORMATION_FILE);
fprintf('Output directory : %s\n', OUTPUT_DIR);

matlabbatch = {};
matlabbatch{1}.spm.util.defs.comp{1}.def = {DEFORMATION_FILE};
matlabbatch{1}.spm.util.defs.out{1}.pull.fnames = {yeo_file};
matlabbatch{1}.spm.util.defs.out{1}.pull.savedir.saveusr = {OUTPUT_DIR};
matlabbatch{1}.spm.util.defs.out{1}.pull.interp = 0;
matlabbatch{1}.spm.util.defs.out{1}.pull.mask = 0;
matlabbatch{1}.spm.util.defs.out{1}.pull.fwhm = [0 0 0];
matlabbatch{1}.spm.util.defs.out{1}.pull.prefix = 'sst_';

spm_jobman('run', matlabbatch);

[~, name, ext] = fileparts(yeo_file);
expected_output = fullfile(OUTPUT_DIR, ['sst_' name ext]);
if ~exist(expected_output, 'file')
    error('SPM completed but the expected SST Yeo label map was not found: %s', expected_output);
end
fprintf('OK: %s\n', expected_output);


function yeo_file = resolve_yeo7_liberal_mask(yeo_dir)
% Resolve the official file name, accepting either .nii or .nii.gz.
base = 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask';
nii_file = fullfile(yeo_dir, [base '.nii']);
gz_file = fullfile(yeo_dir, [base '.nii.gz']);

if exist(nii_file, 'file')
    yeo_file = nii_file;
    return
end
if exist(gz_file, 'file')
    gunzip(gz_file, yeo_dir);  % retain archive; create .nii for SPM
    if exist(nii_file, 'file')
        yeo_file = nii_file;
        return
    end
end
error(['Could not find the Yeo 7-network LiberalMask volume. Expected one of:\n' ...
    '  %s\n  %s'], nii_file, gz_file);
end
