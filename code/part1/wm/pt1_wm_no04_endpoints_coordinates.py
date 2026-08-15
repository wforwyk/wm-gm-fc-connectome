#!/usr/bin/env python3
"""
no4_endpoints_coordinates.py
=============================
PURPOSE : Extract voxel (ijk) and RAS (mm) coordinates from
          TRACULA endpoint probability density maps via fslstats.

PIPELINE CONTEXT:
  dpath/{tract}/endpt{1,2}.pd.nii.gz
      ↓ fslstats -C (voxel COG)  +  -c (RAS COG)
  → {subj}_{tract}_endpt{1,2}.csv  (tract, i, j, k, x, y, z)

INPUT   : {BASE_DIR}/{subj}/wm/freesurfer/{subj}/dpath/{tract}/endpt{1,2}.pd.nii.gz

OUTPUT  : {BASE_DIR}/{subj}/wm/derived/endpoints/endpt{1,2}/
              {subj}_{tract}_endpt{1,2}.csv
          Columns: tract, i, j, k, x, y, z

NOTE    : Numeric subject IDs only (no sub- prefix) for batch.
          fslstats -C returns intensity-weighted center of gravity
          (single coordinate per endpoint, not all active voxels).
          no7_aal_lookup.py reads endpt.pd.nii.gz directly for full voxel lookup.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "0001", "0002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/path/to/your/project/derivatives/batch")

# ────────────────────────────────────────────────────────────

import subprocess
from pathlib import Path


def run_fslstats(nii_path, flag):
    result = subprocess.run(
        ["fslstats", str(nii_path), flag],
        capture_output=True, text=True
    )
    return result.stdout.strip().split()


def run_subject(subj):
    print(f"\n[{subj}]")

    dpath = Path(BASE_DIR) / subj / "wm" / "freesurfer" / subj / "dpath"
    if not dpath.exists():
        print(f"  ERROR: dpath not found: {dpath}")
        return False

    ep_dirs = {
        1: Path(BASE_DIR) / subj / "wm" / "derived" / "endpoints" / "endpt1",
        2: Path(BASE_DIR) / subj / "wm" / "derived" / "endpoints" / "endpt2",
    }
    for d in ep_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    tract_dirs = sorted([d for d in dpath.iterdir() if d.is_dir()])
    n_ok, n_skip = 0, 0

    for tract_dir in tract_dirs:
        tname = tract_dir.name
        for n in [1, 2]:
            nii = tract_dir / f"endpt{n}.pd.nii.gz"
            out = ep_dirs[n] / f"{subj}_{tname}_endpt{n}.csv"

            if not nii.exists():
                n_skip += 1
                continue

            vox = run_fslstats(nii, "-C")  # i j k
            ras = run_fslstats(nii, "-c")  # x y z

            if len(vox) < 3 or len(ras) < 3:
                print(f"  WARN: fslstats returned unexpected output for {tname} endpt{n}")
                n_skip += 1
                continue

            with open(out, "w") as f:
                f.write("tract,i,j,k,x,y,z\n")
                f.write(f"{tname},{vox[0]},{vox[1]},{vox[2]},"
                        f"{ras[0]},{ras[1]},{ras[2]}\n")
            n_ok += 1

    print(f"  OK: {n_ok}  SKIP: {n_skip}")
    return True


def main():
    print("=" * 52)
    print("  no4_endpoints_coordinates.py")
    print(f"  Subjects : {len(SUBJECTS)}")
    print("=" * 52)

    failed = []
    for subj in SUBJECTS:
        try:
            ok = run_subject(subj)
            if not ok:
                failed.append(subj)
        except Exception as e:
            print(f"  ERROR [{subj}]: {e}")
            failed.append(subj)

    print("\n" + "=" * 52)
    if not failed:
        print("  All done. No failures.")
    else:
        print(f"  FAILED ({len(failed)}):")
        for f in failed:
            print(f"    {f}")
    print("=" * 52)


if __name__ == "__main__":
    main()
