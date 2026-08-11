"""
=========================================================================================
Script Name: aim2_endpoint_specificity_extractor.py

Overview:
This script performs the voxel-level analysis for Aim 2 of the WM-GM-FC Connectome 
project. It tests the hypothesis that grey matter (GM) voxels located at the endpoints 
of white matter (WM) tracts exhibit functional connectivity (FC) patterns that reflect 
their distally connected regions, overriding local physical proximity.

Purpose:
To extract the Endpoint Specificity Index (ESI) for each subject and tract. The ESI 
quantifies the strictness of structure-function coupling in native anatomical space. 
These individual-level metrics serve as the primary features for subsequent group-level 
resilience pattern analyses (Aim 3).

Input:
1. Native fMRI space matrices: R.npz (FC, 7677x7677), W_a.npz (Geodesic distance, 7677x7677)
2. GM Spatial Info: GM_mask_7677.nii.gz (3D volume), AAL_7677_labels.npy (1D AAL assignments)
3. TRACULA outputs: fmri_endpt1.pd.nii, fmri_endpt2.pd.nii (native space probability maps)
4. Tract definition: tract_dictionary.csv (defines source/target AAL regions for each tract)

Output:
A consolidated CSV file (Aim2_ESI_results.csv) containing the Endpoint Specificity Index 
(ESI) for each subject, tract, and AAL region.
=========================================================================================
"""

import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

# =======================================================================================
# USER SETTINGS & GLOBAL VARIABLES
# =======================================================================================
PROJECT_DIR = "/bml/projects/06_resilience/projects/06-12_wm-gm-fc-connectome"
DERIVATIVES_DIR = os.path.join(PROJECT_DIR, "derivatives")
TRACULA_DIR = os.path.join(DERIVATIVES_DIR, "batch")

# Percentile threshold for binarizing TRACULA probability distribution (pd) maps
# e.g., 5 means the top 5% of valid probability values will be classified as endpoints
PD_TOP_PERCENTILE = 5.0 

# Geodesic distance threshold (in mm) to define "physically adjacent" local voxels
LOCAL_DIST_MM = 30.0

# Dictionary mapping file (User must ensure this CSV exists with columns: tract, region1, region2)
TRACT_DICT_PATH = os.path.join(PROJECT_DIR, "code", "tract_dictionary.csv")

# Output path for the final Aim 2 results
OUTPUT_CSV = os.path.join(DERIVATIVES_DIR, "Aim2_ESI_results.csv")

# Subject identification (Assuming subjects start with '10')
sub_list = [d for d in os.listdir(TRACULA_DIR) 
            if os.path.isdir(os.path.join(TRACULA_DIR, d)) and d.startswith("10")]
sub_list.sort()


# =======================================================================================
# HELPER FUNCTIONS
# =======================================================================================
def extract_endpoint_mask(pd_nifti_path, top_percent):
    """
    Binarizes a TRACULA probability distribution map using a relative percentile threshold.
    """
    if not os.path.exists(pd_nifti_path):
        return None
        
    img = nib.load(pd_nifti_path)
    pd_data = img.get_fdata()
    
    valid_pd_values = pd_data[pd_data > 0]
    
    if len(valid_pd_values) == 0:
        return np.zeros_like(pd_data, dtype=bool)
    
    threshold = np.percentile(valid_pd_values, 100 - top_percent)
    binary_mask = pd_data >= threshold
    
    return binary_mask

def map_3d_to_1d_indices(binary_mask_3d, gm_voxel_coords):
    """
    Maps a 3D binary mask to the 1D 7677-length index array.
    gm_voxel_coords should be a tuple of arrays representing the X, Y, Z coordinates
    of the 7677 GM voxels in the native fMRI space.
    """
    if binary_mask_3d is None:
        return np.array([], dtype=int)
        
    # Extract the boolean values at the specific 7677 GM coordinates
    # Returns a 1D boolean array of length 7677
    is_endpoint_1d = binary_mask_3d[gm_voxel_coords[0], gm_voxel_coords[1], gm_voxel_coords[2]]
    
    # Return the indices where the value is True
    return np.where(is_endpoint_1d)[0]


# =======================================================================================
# MAIN PROCESSING LOOP
# =======================================================================================
def main():
    print(f"Starting Aim 2 processing for {len(sub_list)} subjects...")
    
    # Load tract definitions (Assuming columns: 'tract', 'region1_aal', 'region2_aal')
    if os.path.exists(TRACT_DICT_PATH):
        tract_df = pd.read_csv(TRACT_DICT_PATH)
    else:
        raise FileNotFoundError(f"Tract dictionary not found at {TRACT_DICT_PATH}. Please create this mapping.")
    
    esi_records = []
    
    for sub in sub_list:
        print(f"Processing Subject: {sub}")
        
        sub_fc_dir = os.path.join(DERIVATIVES_DIR, sub, "func")
        sub_tracula_dpath = os.path.join(TRACULA_DIR, sub, "wm", "freesurfer", sub, "dpath")
        
        # Paths to native space matrices and masks
        fc_path = os.path.join(sub_fc_dir, "R.npz")
        dist_path = os.path.join(sub_fc_dir, "W_a.npz")
        gm_mask_path = os.path.join(sub_fc_dir, "GM_mask_7677.nii.gz") # Adjust name as necessary
        aal_labels_path = os.path.join(sub_fc_dir, "AAL_7677_labels.npy") # Adjust name as necessary
        
        if not all(map(os.path.exists, [fc_path, dist_path, gm_mask_path, aal_labels_path])):
            print(f"  Missing fundamental matrices for {sub}. Skipping.")
            continue
            
        # Load matrices
        FC_matrix = np.load(fc_path)['arr_0']
        Dist_matrix = np.load(dist_path)['arr_0']
        aal_labels = np.load(aal_labels_path)
        
        # Get 3D coordinates for the 7677 GM voxels
        gm_img = nib.load(gm_mask_path)
        gm_data = gm_img.get_fdata()
        gm_coords = np.where(gm_data > 0)
        
        # Iterate over tracts defined in the dictionary
        for _, row in tract_df.iterrows():
            tract_name = row['tract']
            reg1_id = row['region1_aal']
            reg2_id = row['region2_aal']
            
            tract_dir = os.path.join(sub_tracula_dpath, tract_name)
            if not os.path.exists(tract_dir):
                continue
                
            # Extract 3D endpoint masks
            ep1_3d = extract_endpoint_mask(os.path.join(tract_dir, "fmri_endpt1.pd.nii"), PD_TOP_PERCENTILE)
            ep2_3d = extract_endpoint_mask(os.path.join(tract_dir, "fmri_endpt2.pd.nii"), PD_TOP_PERCENTILE)
            
            # Map 3D endpoints to 1D 7677 indices
            ep1_idx = map_3d_to_1d_indices(ep1_3d, gm_coords)
            ep2_idx = map_3d_to_1d_indices(ep2_3d, gm_coords)
            
            # Analyze Region 1 (Endpoint 1 -> Target Region 2)
            process_endpoint_specificity(sub, tract_name, reg1_id, reg2_id, ep1_idx, aal_labels, FC_matrix, Dist_matrix, esi_records)
            
            # Analyze Region 2 (Endpoint 2 -> Target Region 1)
            process_endpoint_specificity(sub, tract_name, reg2_id, reg1_id, ep2_idx, aal_labels, FC_matrix, Dist_matrix, esi_records)

    # Save final output
    final_df = pd.DataFrame(esi_records)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nProcessing complete. Results saved to {OUTPUT_CSV}")

def process_endpoint_specificity(sub, tract_name, source_reg, target_reg, ep_indices, aal_labels, fc_mat, dist_mat, records_list):
    """
    Computes the ESI for a specific endpoint region and appends it to the records list.
    """
    # 1. Identify all voxels belonging to the source region
    source_reg_mask = (aal_labels == source_reg)
    source_all_idx = np.where(source_reg_mask)[0]
    
    if len(source_all_idx) == 0:
        return
        
    # 2. Separate voxels into Endpoints and Non-endpoints
    # Using np.intersect1d and np.setdiff1d for clean index segregation
    ep_voxels = np.intersect1d(source_all_idx, ep_indices)
    nep_voxels = np.setdiff1d(source_all_idx, ep_indices)
    
    if len(ep_voxels) == 0 or len(nep_voxels) == 0:
        return
        
    # 3. Identify target region voxels
    target_reg_mask = (aal_labels == target_reg)
    target_voxels = np.where(target_reg_mask)[0]
    
    if len(target_voxels) == 0:
        return
        
    # 4. Calculate FC from Endpoint voxels to distant Target region
    # Using np.ix_ to slice the specific cross-section of the FC matrix
    fc_ep_to_target = np.nanmean(fc_mat[np.ix_(ep_voxels, target_voxels)])
    
    # 5. Calculate FC from Non-endpoint voxels to local physically adjacent voxels
    # Create a boolean mask of all voxels within LOCAL_DIST_MM of the non-endpoints
    local_dist_mask = (dist_mat[np.ix_(nep_voxels, np.arange(fc_mat.shape[1]))] < LOCAL_DIST_MM)
    
    # Extract FC values where distance condition is met
    nep_fc_subset = fc_mat[np.ix_(nep_voxels, np.arange(fc_mat.shape[1]))]
    fc_nep_to_local = np.nanmean(nep_fc_subset[local_dist_mask])
    
    # 6. Calculate Endpoint Specificity Index (ESI)
    esi_score = fc_ep_to_target - fc_nep_to_local
    
    records_list.append({
        'subject': sub,
        'tract': tract_name,
        'source_region_aal': source_reg,
        'target_region_aal': target_reg,
        'ep_voxel_count': len(ep_voxels),
        'nep_voxel_count': len(nep_voxels),
        'fc_ep_to_target': fc_ep_to_target,
        'fc_nep_to_local': fc_nep_to_local,
        'esi_score': esi_score
    })

if __name__ == "__main__":
    main()