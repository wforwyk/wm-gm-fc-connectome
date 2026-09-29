# Manual template preparation: MNI AAL to SST space

This one-time manual step prepares the AAL label image used by the automated
template scripts. Obtain the atlas and template files separately under their
applicable licences.

## Required inputs

- An AAL label image exported from DSI Studio with its paired exported LUT (`all_ROIs_AAL.txt`)
- Your MNI-to-SST inverse deformation field
- Your DARTEL flow fields, one per subject
- SPM12

Store the shared assets in locations you choose, for example:

```text
/path/to/your/project/derivatives/template/
  AAL3/
  sst_atlas/                    # any name is acceptable; configure the scripts
    sst_all_ROIs_AAL.nii
    <your MNI-to-SST inverse deformation field>
```

## Procedure

1. Export the selected AAL regions as a single label image. Preserve integer
   region IDs; do not convert it into a probabilistic image. Export the LUT from the
   same DSI Studio image. Use this LUT for WM no07/no11 and Part 2 no01.
   The original AAL3 LUT is only a name/category crosswalk reference: its numeric
   IDs must never be used directly with the exported/resampled label images.
2. In SPM, apply your MNI-to-SST inverse deformation to obtain
   `sst_all_ROIs_AAL.nii` in SST space. Use nearest-neighbour interpolation
   when resampling the discrete label atlas.
3. Verify the SST-space AAL image visually against the SST reference and
   confirm that its labels remain integers.
4. Configure `pt1_template_no00_copy_u_rc1.sh` with the location, session,
   and filename token of your own DARTEL flow fields.
5. Set the identical `DARTEL_TEMPLATE_ID` and `SOURCE_SESSION` in template,
   WM, and Part 2 scripts that read the derived AAL images.
6. Run `pt1_template_no02_SST_to_sub.m` and then
   `pt1_template_no03_coregister_to_dwi_fmri.m`.

The automated template scripts expect the flow-field naming pattern
`u_rc1sub-<numeric-id>_<session>_T1w_<DARTEL_TEMPLATE_ID>.nii`. If your flow
fields have another form, adjust the filename construction in template steps
00–04 and keep the same resulting name in every downstream AAL reader.

## Mandatory quality control

- Check the fMRI-space AAL image against the fMRI reference and the DWI-space
  AAL image against the DWI reference.
- Before running endpoint reslicing, visually verify DWI-to-fMRI anatomical
  alignment. Matching image dimensions or voxel sizes alone is not enough.

DARTEL flow fields are generated outside this package during cohort T1 template
construction. Step no00 copies existing fields; it does not estimate them.
