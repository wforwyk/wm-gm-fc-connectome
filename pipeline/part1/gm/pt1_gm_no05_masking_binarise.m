%% =========================================================
%  pt1_gm_no05_masking_binarise.m
%  ---------------------------------------------------------
%  Step 1: Masking by the grey-matter segmentation (prefix 'm')
%  Step 2: Binarisation of the masked series      (prefix 'bm')
%
%  Retain voxels with valid values across the full time series and reject
%  voxels that are zero at every time point.
%
%  INPUT   {BATCH_DIR}/{subj}/gm/func/derived/denoised_detrended_rest_{subj}.nii
%          {BATCH_DIR}/{subj}/gm/anat/segmentation/c1{subj}*.nii
%  OUTPUT  {BATCH_DIR}/{subj}/gm/func/derived/mdenoised_detrended_rest_{subj}.nii
%          {BATCH_DIR}/{subj}/gm/func/derived/bmdenoised_detrended_rest_{subj}.nii
%          {BATCH_DIR}/{subj}/gm/func/derived/gm_no5_mask_qc.csv
%
%  NEXT    pt1_gm_no06_corr_mat_dist.py
%% =========================================================

%% ================= USER SETTINGS =================
BATCH_DIR  = '/path/to/your/project/derivatives/batch';
SUBJECTS  = {'0001','0002'};   % subject_id must be in ' ' and divided by ,

SPM_ROOT  = '/path/to/your/spm12/';

% Grey-matter probability threshold. Higher values retain fewer voxels.
% Inspect gm_no5_mask_qc.csv and the resulting mask when choosing this value.
C1_THRESHOLD = 0.9;

% Report the connected-component structure of the resulting mask.  A sound
% mask is dominated by one component; heavy fragmentation indicates the
% mask criterion is wrong.
REPORT_COMPONENTS = true;

% --- Batch running -----------------------------------------------------
SUBJECTS_TXT  = '';      % non-empty overrides SUBJECTS; one ID per line
EXCLUDE_SUBJECTS = {}; % Set participant exclusions here.
SKIP_EXISTING = true;    % skip subjects that already have the bm mask
%% =================================================

addpath(SPM_ROOT);
spm('defaults', 'FMRI');

addpath(fileparts(fileparts(mfilename('fullpath'))));
SUBJECTS = resolve_participant_list(SUBJECTS, SUBJECTS_TXT, EXCLUDE_SUBJECTS, BATCH_DIR);

if SKIP_EXISTING
    todo = true(1, numel(SUBJECTS));
    for s = 1:numel(SUBJECTS)
        todo(s) = ~exist(fullfile(BATCH_DIR, SUBJECTS{s}, 'gm', 'func', 'derived', ...
                    ['bmdenoised_detrended_rest_' SUBJECTS{s} '.nii']), 'file');
    end
    if any(~todo)
        fprintf('Skipping %d subject(s) that already have a binarised mask.\n', nnz(~todo));
    end
    SUBJECTS = SUBJECTS(todo);
end
fprintf('Subjects to run: %d\n', numel(SUBJECTS));

failed_subjects = {};
qc = {};

for s = 1:numel(SUBJECTS)
    subj = SUBJECTS{s};
    fprintf('\n[%d/%d] Running: %s\n', s, numel(SUBJECTS), subj);

    seg_dir     = fullfile(BATCH_DIR, subj, 'gm', 'anat', 'segmentation');
    derived_dir = fullfile(BATCH_DIR, subj, 'gm', 'func', 'derived');

    try
        % ---- STEP 1: mask the 4-D series by c1 ------------------------
        fprintf('  [1/2] Masking (c1 > %.2f)...\n', C1_THRESHOLD);

        c1_file   = pick_one(fullfile(seg_dir, ['c1' subj '*.nii']), 'c1 segmentation');
        func_file = fullfile(derived_dir, ['denoised_detrended_rest_' subj '.nii']);
        if ~exist(func_file, 'file')
            error('Missing input: %s', func_file);
        end

        spm_mask(c1_file, func_file, C1_THRESHOLD);

        masked_file = fullfile(derived_dir, ['mdenoised_detrended_rest_' subj '.nii']);
        if ~exist(masked_file, 'file')
            error('spm_mask produced no output: %s', masked_file);
        end

        % ---- STEP 2: binarise -----------------------------------------
        fprintf('  [2/2] Binarising...\n');

        V_all = spm_vol(masked_file);
        img   = spm_read_vols(V_all);
        if ndims(img) ~= 4
            error('Masked file is not 4-D (size %s).', mat2str(size(img)));
        end

        % A voxel is valid when it is finite at EVERY time point and is not
        % identically zero.  This covers both conventions spm_mask may use
        % for out-of-mask voxels (NaN or 0).
        valid = ~any(isnan(img), 4) & any(img ~= 0, 4);
        bY    = double(valid);

        [fpath, fname, fext] = fileparts(masked_file);
        Vout         = V_all(1);
        Vout.fname   = fullfile(fpath, ['b' fname fext]);
        Vout.n       = [1 1];
        Vout.dt      = [spm_type('uint8') spm_platform('bigend')];
        Vout.descrip = sprintf('Binarised GM mask (c1 > %.2f, finite across time)', ...
                               C1_THRESHOLD);
        if exist(Vout.fname, 'file'), delete(Vout.fname); end
        spm_write_vol(Vout, bY);

        n_gm = nnz(valid);
        zooms = sqrt(sum(Vout.mat(1:3,1:3).^2, 1));
        vol_cm3 = n_gm * prod(zooms) / 1000;
        fprintf('    n_gm = %d voxels  (%.1f cm^3 at %.2f x %.2f x %.2f mm)\n', ...
                n_gm, vol_cm3, zooms(1), zooms(2), zooms(3));

        % ---- mask integrity check -------------------------------------
        rec = struct('subject', {subj}, 'n_gm', n_gm, 'volume_cm3', vol_cm3, ...
                     'c1_threshold', C1_THRESHOLD, 'n_components', NaN, ...
                     'largest_component_pct', NaN, 'n_isolated_voxels', NaN);

        if REPORT_COMPONENTS
            % bwconncomp needs the Image Processing Toolbox.  If it is not
            % licensed, skip this report: gm_no6 performs the same check
            % with scipy and is the authoritative one.
            try
                CC = bwconncomp(valid, 26);
                szs = cellfun(@numel, CC.PixelIdxList);
                rec.n_components          = CC.NumObjects;
                rec.largest_component_pct = 100 * max(szs) / n_gm;
                rec.n_isolated_voxels     = nnz(szs == 1);
                fprintf('    components = %d | largest = %.1f%% | isolated voxels = %d\n', ...
                        rec.n_components, rec.largest_component_pct, rec.n_isolated_voxels);
                if rec.largest_component_pct < 90
                    warning(['Mask is fragmented for %s (largest component %.1f%%). ' ...
                             'Expect unreachable voxels in the Dijkstra distance matrix.'], ...
                             subj, rec.largest_component_pct);
                end
            catch cc_err
                fprintf(['    component report skipped (%s).\n' ...
                         '    gm_no6 will report components instead.\n'], cc_err.message);
            end
        end

        qc{end+1} = rec; %#ok<SAGROW>
        fprintf('[OK] %s\n', subj);

    catch err
        fprintf('[FAILED] %s\n', subj);
        fprintf('  Error: %s\n', err.message);
        failed_subjects{end+1} = subj; %#ok<SAGROW>
    end
end

%% ================= SUMMARY =================
if ~isempty(qc)
    T = struct2table([qc{:}]);
    qc_path = fullfile(BATCH_DIR, 'gm_no5_mask_qc.csv');
    % Append rather than overwrite, so a resumed batch keeps earlier rows.
    if exist(qc_path, 'file')
        Told = readtable(qc_path, 'TextType', 'string');
        Told = Told(~ismember(string(Told.subject), string(T.subject)), :);
        try
            T = [Told; T];
        catch
            warning('Existing QC file has a different layout; overwriting.');
        end
    end
    writetable(T, qc_path);
    fprintf('\nQC table written: %s\n', qc_path);
    disp(T);
end

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


function fp = pick_one(glob_pattern, label)
    dd = dir(glob_pattern);
    if isempty(dd), error('Missing %s. Pattern: %s', label, glob_pattern); end
    [~, idx] = max([dd.datenum]);
    fp = fullfile(dd(idx).folder, dd(idx).name);
end
