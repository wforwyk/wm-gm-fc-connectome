% =========================================================================
% gm_no1_2_preprocessing.m
% -------------------------------------------------------------------------
% SPM fMRI Preprocessing Pipeline
% Pipeline:
%   Job 1 : Realign: Estimate & Reslice
%   Job 2 : Segment (bias field - based on mean func)
%   Job 3 : Apply Bias Field (call_matlab)
%   Job 4 : Slice Timing Correction
%   Job 5 : Coregister: Estimate (anat -> func)
%   Job 6 : Segment (coregistered structural)
% =========================================================================

%% ================= USER SETTINGS =================
BASE_DIR  = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/batch';
SUBJECTS  = {'3003'};

N_VOLS    = 240;
SPM_ROOT  = '/opt/spm/';
TPM_PATH  = '/opt/spm/tpm/TPM.nii';
BIAS_FUN  = '/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/code/brainworld_scripts_v3_new_revised/gm/apply_bias_field_image.m';
MATLAB_ROOT = matlabroot;   

% Slice Timing parameters
ST_NSLICES  = 32;
ST_TR       = 2;
ST_TA       = 1.9375;
ST_SO       = [2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 ...
               1 3 5 7  9 11 13 15 17 19 21 23 25 27 29 31];
ST_REFSLICE = 2;

%% ================= SPM INITIALIZATION =================
addpath(SPM_ROOT);
spm('defaults', 'FMRI');
spm_jobman('initcfg');

%% ================= MAIN LOOP =================
failed_subjects = {};

ngaus_vals  = [1 1 2 3 4 2];
native_vals = {[1 0],[1 0],[1 0],[1 0],[1 0],[0 0]};

for s = 1:length(SUBJECTS)
    clear matlabbatch;  
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, length(SUBJECTS), subj);

    func_file = fullfile(BASE_DIR, subj, 'gm', 'func', 'preprocessing', [subj '_rest.nii']);
    anat_file = fullfile(BASE_DIR, subj, 'gm', 'anat', 'segmentation',  [subj '_t1.nii']);


    if ~exist(func_file, 'file')
        fprintf('[SKIP] func file not found: %s\n', func_file);
        failed_subjects{end+1} = subj;
        continue;
    end
    if ~exist(anat_file, 'file')
        fprintf('[SKIP] anat file not found: %s\n', anat_file);
        failed_subjects{end+1} = subj;
        continue;
    end

    % Build func scans cell array
    func_scans = cell(N_VOLS, 1);
    for v = 1:N_VOLS
        func_scans{v} = sprintf('%s,%d', func_file, v);
    end

    try
        % ------------------------------------------------------------------
        % Job 1: Realign Estimate & Reslice
        % ------------------------------------------------------------------
        matlabbatch{1}.spm.spatial.realign.estwrite.data              = {func_scans};
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality  = 0.9;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep      = 4;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm     = 5;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm      = 0;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp   = 2;
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap     = [0 0 0];
        matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight   = '';
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which    = [2 1];
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp   = 4;
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap     = [0 0 0];
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask     = 1;
        matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix   = 'r';

        % ------------------------------------------------------------------
        % Job 2: Segment (mean func - for bias field extraction)
        % ------------------------------------------------------------------
        matlabbatch{2}.spm.spatial.preproc.channel.vols(1)       = cfg_dep('Realign: Estimate & Reslice: Mean Image', ...
            substruct('.','val','{}',{1},'.','val','{}',{1},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('.','rmean'));
        matlabbatch{2}.spm.spatial.preproc.channel.biasreg        = 0.001;
        matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm       = 60;
        matlabbatch{2}.spm.spatial.preproc.channel.write          = [1 0];
        for t = 1:6
            matlabbatch{2}.spm.spatial.preproc.tissue(t).tpm     = {sprintf('%s,%d', TPM_PATH, t)};
            matlabbatch{2}.spm.spatial.preproc.tissue(t).ngaus   = ngaus_vals(t);
            matlabbatch{2}.spm.spatial.preproc.tissue(t).native  = native_vals{t};
            matlabbatch{2}.spm.spatial.preproc.tissue(t).warped  = [0 0];
        end
        matlabbatch{2}.spm.spatial.preproc.warp.mrf               = 1;
        matlabbatch{2}.spm.spatial.preproc.warp.cleanup           = 1;
        matlabbatch{2}.spm.spatial.preproc.warp.reg               = [0 0.001 0.5 0.05 0.2];
        matlabbatch{2}.spm.spatial.preproc.warp.affreg            = 'eastern';
        matlabbatch{2}.spm.spatial.preproc.warp.fwhm              = 0;
        matlabbatch{2}.spm.spatial.preproc.warp.samp              = 3;
        matlabbatch{2}.spm.spatial.preproc.warp.write             = [0 0];
        matlabbatch{2}.spm.spatial.preproc.warp.vox               = NaN;
        matlabbatch{2}.spm.spatial.preproc.warp.bb                = [NaN NaN NaN; NaN NaN NaN];

        % ------------------------------------------------------------------
        % Job 3: Apply Bias Field (call_matlab)
        % ------------------------------------------------------------------
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.inputs{1}.images(1) = cfg_dep('Segment: Bias Field (1)', ...
            substruct('.','val','{}',{2},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('()',{1},'.','channel','()',{1},'.','biasfield','()',{':'}));
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.outputs.filter.nifti                   = 1;
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.matlabroot                         = MATLAB_ROOT;  
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.separator                          = '/';
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.sentinel                           = '@';
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.function_handle.function           = 'apply_bias_field_image';
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.function_handle.type               = 'simple';
        matlabbatch{3}.cfg_basicio.run_ops.call_matlab.fun.function_handle.file               = BIAS_FUN;

        % ------------------------------------------------------------------
        % Job 4: Slice Timing Correction
        % ------------------------------------------------------------------
        matlabbatch{4}.spm.temporal.st.scans{1}(1) = cfg_dep('Call MATLAB function: Call MATLAB: output 1 - filter nifti', ...
            substruct('.','val','{}',{3},'.','val','{}',{1},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('()',{1},'.','outputs','()',{':'}));
        matlabbatch{4}.spm.temporal.st.nslices  = ST_NSLICES;
        matlabbatch{4}.spm.temporal.st.tr       = ST_TR;
        matlabbatch{4}.spm.temporal.st.ta       = ST_TA;
        matlabbatch{4}.spm.temporal.st.so       = ST_SO;
        matlabbatch{4}.spm.temporal.st.refslice = ST_REFSLICE;
        matlabbatch{4}.spm.temporal.st.prefix   = 'a';

        % ------------------------------------------------------------------
        % Job 5: Coregister Estimate (anat -> func)
        % ------------------------------------------------------------------
        matlabbatch{5}.spm.spatial.coreg.estimate.ref(1) = cfg_dep('Slice Timing: Slice Timing Corr. Images (Sess 1)', ...
            substruct('.','val','{}',{4},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('()',{1}));
        matlabbatch{5}.spm.spatial.coreg.estimate.source              = {[anat_file ',1']};
        matlabbatch{5}.spm.spatial.coreg.estimate.other               = {''};
        matlabbatch{5}.spm.spatial.coreg.estimate.eoptions.cost_fun   = 'nmi';
        matlabbatch{5}.spm.spatial.coreg.estimate.eoptions.sep        = [4 2];
        matlabbatch{5}.spm.spatial.coreg.estimate.eoptions.tol        = [0.02 0.02 0.02 0.001 0.001 0.001 ...
                                                                          0.01  0.01  0.01  0.001 0.001 0.001];
        matlabbatch{5}.spm.spatial.coreg.estimate.eoptions.fwhm       = [7 7];

        % ------------------------------------------------------------------
        % Job 6: Segment (coregistered structural)
        % ------------------------------------------------------------------
        matlabbatch{6}.spm.spatial.preproc.channel.vols(1)       = cfg_dep('Coregister: Estimate: Coregistered Images', ...
            substruct('.','val','{}',{5},'.','val','{}',{1},'.','val','{}',{1},'.','val','{}',{1}), ...
            substruct('.','cfiles'));
        matlabbatch{6}.spm.spatial.preproc.channel.biasreg        = 0.001;
        matlabbatch{6}.spm.spatial.preproc.channel.biasfwhm       = 60;
        matlabbatch{6}.spm.spatial.preproc.channel.write          = [1 0];
        for t = 1:6
            matlabbatch{6}.spm.spatial.preproc.tissue(t).tpm     = {sprintf('%s,%d', TPM_PATH, t)};
            matlabbatch{6}.spm.spatial.preproc.tissue(t).ngaus   = ngaus_vals(t);
            matlabbatch{6}.spm.spatial.preproc.tissue(t).native  = native_vals{t};
            matlabbatch{6}.spm.spatial.preproc.tissue(t).warped  = [0 0];
        end
        matlabbatch{6}.spm.spatial.preproc.warp.mrf               = 1;
        matlabbatch{6}.spm.spatial.preproc.warp.cleanup           = 1;
        matlabbatch{6}.spm.spatial.preproc.warp.reg               = [0 0.001 0.5 0.05 0.2];
        matlabbatch{6}.spm.spatial.preproc.warp.affreg            = 'eastern';
        matlabbatch{6}.spm.spatial.preproc.warp.fwhm              = 0;
        matlabbatch{6}.spm.spatial.preproc.warp.samp              = 3;
        matlabbatch{6}.spm.spatial.preproc.warp.write             = [0 0];
        matlabbatch{6}.spm.spatial.preproc.warp.vox               = NaN;
        matlabbatch{6}.spm.spatial.preproc.warp.bb                = [NaN NaN NaN; NaN NaN NaN];

        % ------------------------------------------------------------------
        % Run
        % ------------------------------------------------------------------
        spm_jobman('run', matlabbatch);
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