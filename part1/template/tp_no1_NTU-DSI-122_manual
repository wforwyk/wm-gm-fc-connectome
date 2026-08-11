## Template Preparation : MNI ----> SST

This stage prepares atlas resources that are reused across all subjects.

### Purpose

* Create ROI-labelled AAL masks suitable for subject-space WM analysis
* Preserve ROI name–label mapping through spatial transformations

### Key Steps

1. Load **NTU-DSI-122** and **AAL atlas** in DSI Studio and click 'reconstruct'

2. Load ROI_MNI_V7_1mm.nii as 'Load regions'

   * The ROI_MNI_V7_1mm.txt files must be edited remaining only ROI numbers and regions names, so that DSI studio can load the ROI region names accordingly when loading the images.

3. Save the Loaded ROI images by 'export'

   * output file : all_ROIs_AAL.nii
   * you can save the each ROI region separately and all together 


4. Use existing DARTEL flow fields to enable:

   MNI ----> SST   ----> subject space
      :iy_NCKU_336Ss_6 
                     :u_rc1sub-xxxx_ses-1_T1w_NCKU_336Ss.nii 

   Since AAL is already standard template, its space is already aligned with MNI.

   * MNI → SST transformation. Warp AAL from NTU-DSI-122 space to SST

      Launch SPM in MATLAB and load batch script 'NTU-DSI_to_SST.mat' [deformation @spm]
   
      * using 'iy_NCKU_336Ss_6.nii'
      * output file : sst_all_ROIs_AAL.nii


This stage is performed **once** which is independent of subject-wise loops, and reused across all subjects. Please save it at the same level of the batch folder.
e.g. 'iy_NCKU_336Ss_6.nii' is saved at /bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome/derivatives/template/NCKU.




### step by step manual for the method wm-roi-based-FA in subject space ###

1. Prepare template atlas and subject data
	- standard template: NTU-DSI-122.nii.gz
	- /bml/projects/08_brainworld/projects/08-05_wm-gm-fc-connectome/derivatives/wm-roi-method/template/NTU-DSI-122

	- ROI template: AAL3
	- /bml/projects/08_brainworld/projects/08-05_wm-gm-fc-connectome/derivatives/wm-roi-method/template/AAL3/ROI_MNI_V7_1mm.nii

2. Load NTU-DSI-122 to DSI studio and reconstruct

3. Load ROI_MNI_V7_1mm.nii as 'Load regions'
	- The ROI_MNI_V7_1mm.txt files must be edited remaining only ROI numbers and regions names, so that DSI studio can load the ROI region names accordingly when loading the images.

4. Save the Loaded ROI images (export)
	- output file : all_ROIs_AAL.nii
	- you can save the each ROI region separately and all together 

5. NTU-DSI-122 --> SST 
	- using 'iy_NCKU_336Ss_6.nii'
	- /bml/projects/08_brainworld/projects/08-05_wm-gm-fc-connectome/derivatives/wm-roi-method/template/NCKU
	- pre-saved batch script 'NTU-DSI_to_SST.mat' [deformation @spm]
	- /bml/projects/08_brainworld/projects/08-05_wm-gm-fc-connectome/code/spm/

	- output file : sst_all_ROIs_AAL.nii

6. SST (i.e. sst_all_ROIs_AAL.nii) --> each subject space
	- using 'u_rc1sub-xxxx_ses-1_T1w_NCKU_336Ss.nii' (from Chun-Wei's project .56/06_resilience/06-04..)
	- pre-saved batch script 'SST_to_sub.mat' [create inversed warp @ spm]
	- /bml/projects/08_brainworld/projects/08-05_wm-gm-fc-connectome/code/spm/
	
	- output file: wsst_all_ROIs_AAL_u_rc1sub-xxxx_ses-1_T1w_NCKU_336Ss.nii


