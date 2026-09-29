#!/usr/bin/env python3
"""Build AAL category and adjacency tables in the DSI Studio export label namespace.

Map canonical AAL categories and adjacency to exported atlas IDs.
Names link export IDs to canonical AAL3 IDs; category rules use canonical IDs.
BORDER_MAT must contain canonical IDs. Its original six-neighbour geometry is
preserved, then pair IDs are translated into the export namespace.
Run once at atlas level, not once per participant. Downstream scripts must use these export-numbered tables with the matching atlas.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# USER SETTINGS
# =============================================================================
AAL_LUT = '/path/to/your/project/derivatives/template/DSIstudio/all_ROIs_AAL.txt'
ORIGINAL_AAL_TXT = '/path/to/your/project/derivatives/template/AAL3/AAL3v1.nii.txt'
BORDER_MAT = '/path/to/your/project/derivatives/template/AAL3/ROI_MNI_V7_1mm_Border.mat'
OUTPUT_DIR = '/path/to/your/project/derivatives/template/AAL3'
REGION_OUTPUT = 'aal3_region_category.csv'
ADJACENCY_OUTPUT = 'aal3_adjacent_pairs.csv'
# =============================================================================


def read_lut(path):
    rows = []
    with open(path, encoding='utf-8-sig') as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith('#'):
                continue
            if len(parts) < 2:
                raise ValueError(f'Malformed AAL_LUT row in {path}: {line!r}')
            rows.append({'roi_id': int(parts[0]), 'roi_name': parts[1]})
    frame = pd.DataFrame(rows, columns=['roi_id', 'roi_name'])
    if frame.empty or frame.roi_id.le(0).any():
        raise ValueError(f'Empty AAL_LUT or nonpositive ROI ID: {path}')
    if frame.roi_id.duplicated().any() or frame.roi_name.duplicated().any():
        raise ValueError(f'Duplicate ROI IDs or names: {path}')
    return frame


def assign_category(roi_id):
    if roi_id in range(41, 47):
        return 'subcortical_limbic'
    elif roi_id in range(75, 81):
        return 'subcortical_basal'
    elif roi_id in [81, 82]:
        return 'thalamus_generic'
    elif roi_id in range(83, 95):
        return 'cortical'
    elif roi_id in range(1, 41):
        return 'cortical'
    elif roi_id in range(47, 75):
        return 'cortical'
    elif roi_id in range(95, 121):
        return 'cerebellum'
    elif roi_id in range(121, 151):
        return 'thalamic_nuclei'
    elif roi_id in range(151, 157):
        return 'acc_subdivision'
    elif roi_id in [157, 158]:
        return 'n_acc'
    elif roi_id in range(159, 171):
        return 'brainstem'
    else:
        return 'unknown'

INCLUDE_CATEGORIES = {
    'cortical', 'subcortical_limbic', 'thalamus_generic', 'thalamic_nuclei', 'acc_subdivision'
}


def build_regions(export, canonical):
    by_name = canonical.set_index('roi_name').roi_id
    original_ids = export.roi_name.map(by_name)
    if original_ids.isna().any():
        raise ValueError(f'Export names missing from canonical AAL_LUT: {export.loc[original_ids.isna(), "roi_name"].tolist()}')
    export_to_original = dict(zip(export.roi_id.astype(int), original_ids.astype(int)))
    original_to_export = {v: k for k, v in export_to_original.items()}
    if len(original_to_export) != len(export):
        raise ValueError('ROI correspondence is not one-to-one')
    result = export.copy()
    result['category'] = original_ids.map(assign_category)
    if result.category.eq('unknown').any():
        raise ValueError('Unclassified canonical ROI ID')
    result['include'] = result.category.isin(INCLUDE_CATEGORIES).astype(int)
    return result, original_to_export


def build_adjacency(border_v, border_xyz, regions, original_to_export):
    values = np.asarray(border_v).reshape(-1)
    coordinates = np.asarray(border_xyz).T
    if coordinates.shape != (len(values), 3):
        raise ValueError('Expected BORDER_XYZ shape (3, N) and BORDER_V length N')
    for name, a in [('BORDER_V', values), ('BORDER_XYZ', coordinates)]:
        if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
            raise ValueError(f'{name} must contain finite integer values')
    values = values.astype(int)
    coordinates = coordinates.astype(int)
    missing = set(values) - set(original_to_export)
    if missing:
        raise ValueError(f'Canonical border IDs absent from export correspondence: {sorted(missing)}. Check BORDER_MAT namespace.')
    # Reject conflicting duplicate coordinates instead of silently keeping the last.
    coord_to_roi = {}
    for xyz, value in zip(coordinates, values):
        key = tuple(xyz)
        if key in coord_to_roi and coord_to_roi[key] != value:
            raise ValueError(f'Conflicting border labels at {key}')
        coord_to_roi[key] = int(value)
    pairs = set()
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for (x,y,z), a in coord_to_roi.items():
        for dx,dy,dz in directions:
            b = coord_to_roi.get((x+dx,y+dy,z+dz))
            if b is not None and a != b:
                pairs.add(tuple(sorted((a,b))))
    export_pairs = {tuple(sorted((original_to_export[a], original_to_export[b]))) for a,b in pairs}
    if len(export_pairs) != len(pairs):
        raise ValueError('ROI remapping changed the number of geometric adjacency pairs')
    lookup = regions.set_index('roi_id')
    rows = []
    for a,b in sorted(export_pairs):
        ra,rb = lookup.loc[a],lookup.loc[b]
        rows.append({'roi_id_a':a, 'roi_name_a':ra.roi_name,
                     'category_a':ra.category, 'include_a':int(ra.include),
                     'roi_id_b':b, 'roi_name_b':rb.roi_name,
                     'category_b':rb.category, 'include_b':int(rb.include),
                     'both_included':int(ra.include == 1 and rb.include == 1)})
    columns = ['roi_id_a','roi_name_a','category_a','include_a',
               'roi_id_b','roi_name_b','category_b','include_b','both_included']
    return pd.DataFrame(rows, columns=columns)


def main():
    import scipy.io
    for path in (AAL_LUT, ORIGINAL_AAL_TXT, BORDER_MAT):
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing input: {path}. Check USER SETTINGS; do not substitute the canonical AAL_LUT for the export AAL_LUT.')
    export, canonical = read_lut(AAL_LUT), read_lut(ORIGINAL_AAL_TXT)
    regions, original_to_export = build_regions(export, canonical)
    mat = scipy.io.loadmat(BORDER_MAT)
    adjacent = build_adjacency(mat['BORDER_V'], mat['BORDER_XYZ'], regions, original_to_export)
    # Validate all inputs and build both tables before writing either output.
    for name in (REGION_OUTPUT, ADJACENCY_OUTPUT):
        if Path(name).name != name or not name.endswith('.csv'):
            raise ValueError('Output filenames must be plain names ending in .csv')
    if REGION_OUTPUT == ADJACENCY_OUTPUT:
        raise ValueError('Output filenames must differ')
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    regions.to_csv(output / REGION_OUTPUT, index=False)
    adjacent.to_csv(output / ADJACENCY_OUTPUT, index=False)
    print(f'Export regions: {len(regions)}; included: {int(regions.include.sum())}')
    print(regions.groupby(['category','include']).size().rename('n').to_string())
    print(f'Geometric adjacency pairs (unchanged by renumbering): {len(adjacent)}')
    print(f'Both included: {int(adjacent.both_included.sum())}')
    print(f'Saved: {output / REGION_OUTPUT}')
    print(f'Saved: {output / ADJACENCY_OUTPUT}')


if __name__ == '__main__':
    main()
