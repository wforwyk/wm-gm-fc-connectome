#!/usr/bin/env python3
"""
no11_wm-gm-fc_3d_v2.py
================
PURPOSE : Generate 3D scatter plots (WM metric x GM geodesic distance x FC)
          and voxelpair CSVs. Loops over multiple subjects.
          One point = one GM voxel pair connected via a WM tract.

BUG FIX (v1 -> v2)
    build_region_groups() used to derive each AAL region's voxel indices
    from `np.where(rc1*.nii > 0)`, then treat the ENUMERATE POSITION within
    that set as if it were the row index into R.npz / W_a.npz. It is not.
    R.npz and W_a.npz are built (gm_no6) from the FINAL cleaned BnB mask
    (c1 > 0.9, finite, var > 0, largest-component filtered) -- a much
    smaller, differently-ordered voxel set than "any voxel with c1 > 0".
    Indexing R[uu_flat, vv_flat] with the old row_idx therefore pulled
    essentially unrelated matrix entries: not the FC/distance between the
    intended voxels. This is independent of any GM pipeline rerun -- it
    would have produced invalid output even against the old R.npz/W_a.npz.

    Fix: build region groups from {subj}_vset.npz's `coords`, which gm_no6
    saves in EXACTLY the row order used to build R (`Y[keep]`) and W
    (Dijkstra source order). enumerate(coords) now gives the true row index.

PIPELINE CONTEXT:
  no4 → endpoints/endpt{1,2}/{subj}_{tract}_endpt{1,2}.csv   (endpoint ijk)
  no6 → {subj}_pathstats.csv                                  (FA/RD/MD/AD)
  step3 → dwi_wsst_all_ROIs_AAL_{subj}.nii                    (DWI AAL atlas)
  step3 → fmri_wsst_all_ROIs_AAL_{subj}.nii                   (fMRI AAL atlas)
  gm_no6 → {subj}_vset.npz                                    (coords, row order of R/W)
  R.npz, W_a.npz                                              (FC, dist)
      ↓ per metric (FA, RD, MD, AD)
  → {subj}_voxelpair_{metric}.csv   (fc, dist, metric, tract, idx_u, idx_v)
  → {subj}_tract_3d_{metric}.png
  → {subj}_tract_3d_{metric}.html   (interactive, if plotly installed)

INPUT   : dwi_aal      — dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii
          fmri_aal     — fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii
          fc           — {subj}_R.npz
          dist         — {subj}_W_a.npz
          vset         — {subj}_vset.npz   (gm/func/derived/, from gm_no6)
          pathstats    — wm/derived/pathstats/{subj}_pathstats.csv
          endpt_dir    — wm/derived/endpoints/
          lut          — AAL3/AAL3v1.nii.txt

OUTPUT  : {OUTDIR}/{subj}_voxelpair_{metric}.csv
          {OUTDIR}/{subj}_tract_3d_{metric}.png
          {OUTDIR}/{subj}_tract_3d_{metric}.html

NOTE    : Numeric subject IDs only (no sub- prefix) for batch2.
          voxelpair CSV (idx_u, idx_v, RD) is required by no12.
          idx_u / idx_v are now true row indices into R.npz / W_a.npz.
          MAX_VOXELS caps voxels sampled per region to avoid memory issues.
"""

# ── USER SETTINGS ───────────────────────────────────────────
SUBJECTS = [
     "1001" , "1002"
] # subject id must be in " " and divided by ,


BASE_DIR = ("/your/base/directory/is/your/root/directory")

LUT = (
    "/bml/projects/06_resilience/projects"
    "/06-12_wm-gm-fc-connectome/derivatives/template/AAL3/AAL3v1.nii.txt"
)

METRICS    = ["FA", "RD", "MD", "AD"]
MAX_VOXELS = 200
ELEV       = 25
AZIM       = -50
DPI        = 300
# ────────────────────────────────────────────────────────────

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import binned_statistic_2d

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_lut(path):
    """
    AAL3v1.nii.txt is whitespace-separated: "<id> <name> [<index>]".
    Most rows carry a trailing index column, a few (e.g. 35/36 Cingulate_Ant,
    81/82 Thalamus) do not.  Only the second field is the region name --
    joining columns 2+ would yield "Precentral_L 1" for most regions and a
    bare name for the rest, i.e. two different name formats in one table.
    Must stay identical to load_lut in pt1_wm_no07: region names are the
    join key between aal_summary.csv and the region groups built here.
    """
    lut = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    lut[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    return lut


def load_endpoint_coords(endpt_dir):
    endpt_dir = Path(endpt_dir)
    tract_coords = {}
    for ep_num in ["endpt1", "endpt2"]:
        ep_dir = endpt_dir / ep_num
        if not ep_dir.exists():
            continue
        for csv_path in sorted(ep_dir.glob("*.csv")):
            fname = csv_path.stem
            parts = fname.split("_")
            for idx, p in enumerate(parts):
                if "endpt" in p:
                    tract_name = "_".join(parts[1:idx])
                    break
            else:
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            row = df.iloc[0]
            try:
                i = int(round(row["i"]))
                j = int(round(row["j"]))
                k = int(round(row["k"]))
            except KeyError:
                continue
            tract_coords.setdefault(tract_name, {})[ep_num] = (i, j, k)
    complete = {t: v for t, v in tract_coords.items()
                if "endpt1" in v and "endpt2" in v}
    print(f"    Tracts with both endpoints: {len(complete)}")
    return complete


def lookup_aal(ijk, dwi_aal_data, lut):
    i, j, k = ijk
    sh = dwi_aal_data.shape
    if not (0 <= i < sh[0] and 0 <= j < sh[1] and 0 <= k < sh[2]):
        return "NA"
    rid = int(dwi_aal_data[i, j, k])
    if rid == 0:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    ni, nj, nk = i+di, j+dj, k+dk
                    if 0<=ni<sh[0] and 0<=nj<sh[1] and 0<=nk<sh[2]:
                        r = int(dwi_aal_data[ni, nj, nk])
                        if r > 0:
                            return lut.get(r, "NA")
        return "NA"
    return lut.get(rid, "NA")


def build_region_groups(fmri_aal_path, vset_path, lut):
    """
    Group R.npz / W_a.npz row indices by AAL region.

    row_idx here MUST be the true row index into R/W. gm_no6 saves
    {subj}_vset.npz with `coords` in exactly the order used to build R
    (Y[keep]) and W (Dijkstra source order), so enumerate(coords) gives
    that index directly -- unlike a fresh threshold of rc1*.nii, which
    is neither the same voxel set nor the same order as R/W's rows.
    """
    aal_data = nib.load(fmri_aal_path).get_fdata().astype(np.int32)
    vset = np.load(vset_path)
    coords = vset["coords"]  # [n x 3], row order matches R.npz / W_a.npz
    sh = aal_data.shape

    groups = {}
    n_oob = 0
    for row_idx, (i, j, k) in enumerate(coords):
        if not (0 <= i < sh[0] and 0 <= j < sh[1] and 0 <= k < sh[2]):
            n_oob += 1
            continue
        aid = int(aal_data[i, j, k])
        name = lut.get(aid, "NA")
        if name != "NA":
            groups.setdefault(name, []).append(row_idx)
    if n_oob:
        print(f"    WARNING: {n_oob} vset voxels fell outside the AAL atlas grid "
              f"-- check that fmri_aal shares the BnB grid.")
    print(f"    R/W voxel set: {len(coords):,}   AAL regions: {len(groups)}")
    return groups


def get_voxel_pairs(reg_u, reg_v, region_groups, R, W, max_vox):
    idx_u = np.array(region_groups.get(reg_u, []))
    idx_v = np.array(region_groups.get(reg_v, []))
    idx_u = idx_u[idx_u < R.shape[0]]
    idx_v = idx_v[idx_v < R.shape[0]]
    if len(idx_u) == 0 or len(idx_v) == 0:
        return (np.array([]), np.array([]),
                np.array([], dtype=int), np.array([], dtype=int))
    if len(idx_u) > max_vox:
        idx_u = np.random.default_rng(42).choice(idx_u, max_vox, replace=False)
    if len(idx_v) > max_vox:
        idx_v = np.random.default_rng(43).choice(idx_v, max_vox, replace=False)
    uu, vv = np.meshgrid(idx_u, idx_v, indexing="ij")
    uu_flat, vv_flat = uu.ravel(), vv.ravel()
    fc_b   = R[uu_flat, vv_flat]
    dist_b = W[uu_flat, vv_flat]
    valid  = np.isfinite(fc_b) & np.isfinite(dist_b) & (dist_b > 0)
    return fc_b[valid], dist_b[valid], uu_flat[valid], vv_flat[valid]


def density_alpha(dist, fc, bins=40, a_min=0.05, a_max=0.6):
    counts, xe, ye, _ = binned_statistic_2d(
        dist, fc, None, statistic="count", bins=bins)
    xi = np.clip(np.searchsorted(xe, dist) - 1, 0, bins-1)
    yi = np.clip(np.searchsorted(ye, fc)   - 1, 0, bins-1)
    cnt = counts[xi, yi].astype(float)
    return (a_max - (a_max-a_min)*(cnt-cnt.min())/(cnt.max()-cnt.min()+1e-9)
            ).clip(a_min, a_max)


def plot_3d(df, subj, metric, outdir, elev, azim):
    if df.empty:
        return
    norm   = mcolors.Normalize(vmin=np.percentile(df["fc"], 1),
                                vmax=np.percentile(df["fc"], 99))
    cmap   = cm.RdBu_r
    colors = cmap(norm(df["fc"].values))
    colors[:, 3] = density_alpha(df["dist"].values, df["fc"].values)

    fig = plt.figure(figsize=(10, 7.5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.scatter(df["dist"], df[metric], df["fc"],
               c=colors, s=1.5, linewidths=0,
               rasterized=True, depthshade=True)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    xx, yy = np.meshgrid(xlim, ylim)
    ax.plot_surface(xx, yy, np.zeros_like(xx),
                    alpha=0.06, color="grey", linewidth=0)
    ax.set_xlabel("GM Geodesic Distance (mm)", fontsize=10, labelpad=10)
    ax.set_ylabel(f"WM {metric}",              fontsize=10, labelpad=10)
    ax.set_zlabel("FC (Pearson r)",            fontsize=10, labelpad=8)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=elev, azim=azim)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.12, aspect=18)
    cbar.set_label("FC (Pearson r)", fontsize=9)
    ax.set_title(
        f"Subject: {subj}\nWM {metric} x GM geodesic dist x FC\n"
        f"n={len(df):,} voxel pairs  {df['tract'].nunique()} tracts",
        fontsize=9, pad=12)
    fig.tight_layout()
    out = Path(outdir) / f"{subj}_tract_3d_{metric}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> PNG: {out.name}")


def plot_3d_interactive(df, subj, metric, outdir):
    if not HAS_PLOTLY or df.empty:
        return
    fig = go.Figure(data=[go.Scatter3d(
        x=df["dist"], y=df[metric], z=df["fc"],
        mode="markers", text=df["tract"],
        hovertemplate=(f"<b>%{{text}}</b><br>GM dist=%{{x:.1f}}mm<br>"
                       f"{metric}=%{{y:.5f}}<br>FC=%{{z:.4f}}<extra></extra>"),
        marker=dict(size=1.5, color=df["fc"], colorscale="RdBu", reversescale=True,
                    colorbar=dict(title="FC (r)"), opacity=0.5,
                    cmin=float(np.percentile(df["fc"], 1)),
                    cmax=float(np.percentile(df["fc"], 99)))
    )])
    fig.update_layout(
        title=f"Subject: {subj}<br>WM {metric} x GM dist x FC   n={len(df):,}",
        scene=dict(xaxis_title="GM Geodesic Distance (mm)",
                   yaxis_title=f"WM {metric}", zaxis_title="FC (Pearson r)"),
        width=1100, height=800)
    out = Path(outdir) / f"{subj}_tract_3d_{metric}.html"
    fig.write_html(str(out))
    print(f"    -> HTML: {out.name}")


def run_subject(subj, lut):
    dwi_aal   = f"{BASE_DIR}/{subj}/gm/derived/dwi_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii"
    fmri_aal  = f"{BASE_DIR}/{subj}/gm/derived/fmri_wsst_all_ROIs_AAL_u_rc1sub-{subj}_ses-1_T1w_NCKU_336Ss.nii"
    fc        = f"{BASE_DIR}/{subj}/gm/func/derived/fc_results/{subj}_R.npz"
    dist      = f"{BASE_DIR}/{subj}/gm/anat/derived/dist_results/{subj}_W_a.npz"
    vset      = f"{BASE_DIR}/{subj}/gm/func/derived/{subj}_vset.npz"
    pathstats = f"{BASE_DIR}/{subj}/wm/derived/pathstats/{subj}_pathstats.csv"
    endpt_dir = f"{BASE_DIR}/{subj}/wm/derived/endpoints"
    outdir    = Path(BASE_DIR) / subj / "wm-gm-fc" / "3d"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{subj}]")
    for p in (fc, dist, vset):
        if not Path(p).exists():
            print(f"  ERROR: missing required input: {p}")
            return

    print("  Loading endpoint coordinates...")
    tract_coords = load_endpoint_coords(endpt_dir)

    print("  Loading DWI-space AAL atlas...", end=" ", flush=True)
    dwi_aal_data = nib.load(dwi_aal).get_fdata().astype(np.int32)
    print(f"shape={dwi_aal_data.shape}")

    tract_aal = {}
    for tract_name, endpoints in tract_coords.items():
        r1 = lookup_aal(endpoints["endpt1"], dwi_aal_data, lut)
        r2 = lookup_aal(endpoints["endpt2"], dwi_aal_data, lut)
        if r1 != "NA" and r2 != "NA":
            tract_aal[tract_name] = {"endpt1": r1, "endpt2": r2}
    print(f"  Tracts with valid AAL: {len(tract_aal)}")
    del dwi_aal_data; gc.collect()

    print("  Building fMRI region groups (from vset.npz, matches R/W row order)...")
    region_groups = build_region_groups(fmri_aal, vset, lut)

    print("  Loading R.npz ...", end=" ", flush=True)
    R = np.load(fc)["arr_0"].astype(np.float32)
    print(f"shape={R.shape}")
    print("  Loading W_a.npz ...", end=" ", flush=True)
    W = np.load(dist)["arr_0"].astype(np.float32)
    print(f"shape={W.shape}")

    path_df = pd.read_csv(pathstats)
    path_df.columns = path_df.columns.str.strip()
    path_lookup = {str(r["tract"]).strip(): r for _, r in path_df.iterrows()}

    fc_list, dist_list, tract_list, idx_u_list, idx_v_list = [], [], [], [], []

    for tract_name, endpoints in tract_aal.items():
        fc_pairs, dist_pairs, iu, iv = get_voxel_pairs(
            endpoints["endpt1"], endpoints["endpt2"],
            region_groups, R, W, MAX_VOXELS)
        if len(fc_pairs) == 0:
            continue
        fc_list.append(fc_pairs)
        dist_list.append(dist_pairs)
        tract_list.append(np.full(len(fc_pairs), tract_name, dtype=object))
        idx_u_list.append(iu)
        idx_v_list.append(iv)

    del R, W; gc.collect()

    if not fc_list:
        print("  No voxel pairs found.")
        return

    all_fc    = np.concatenate(fc_list)
    all_dist  = np.concatenate(dist_list)
    all_tract = np.concatenate(tract_list)
    all_idx_u = np.concatenate(idx_u_list)
    all_idx_v = np.concatenate(idx_v_list)
    print(f"  Total voxel pairs: {len(all_fc):,}")

    for metric in METRICS:
        print(f"\n  [{metric}]")
        metric_col = f"{metric}_Avg"
        tract_metric = {}
        for tract_name in np.unique(all_tract):
            ps = path_lookup.get(tract_name)
            if ps is None:
                continue
            val = ps.get(metric_col, np.nan)
            if not pd.isna(val):
                tract_metric[tract_name] = float(val)

        wm_vals = np.array([tract_metric.get(t, np.nan) for t in all_tract])
        valid   = np.isfinite(wm_vals)

        df = pd.DataFrame({
            "fc"    : all_fc[valid],
            "dist"  : all_dist[valid],
            metric  : wm_vals[valid],
            "tract" : all_tract[valid],
            "idx_u" : all_idx_u[valid],
            "idx_v" : all_idx_v[valid],
        })
        print(f"    {len(df):,} pairs from {df['tract'].nunique()} tracts")
        if df.empty:
            continue

        out_csv = outdir / f"{subj}_voxelpair_{metric}.csv"
        df.to_csv(out_csv, index=False)
        print(f"    -> CSV: {out_csv.name}")
        plot_3d(df, subj, metric, str(outdir), ELEV, AZIM)
        plot_3d_interactive(df, subj, metric, str(outdir))

    print(f"  Done -> {outdir.resolve()}")


def main():
    print("=" * 52)
    print("  no11_wm-gm-fc_3d_v2.py")
    print(f"  Subjects : {len(SUBJECTS)}")
    print(f"  Metrics  : {METRICS}")
    print("=" * 52)

    lut = load_lut(LUT)

    failed = []
    for subj in SUBJECTS:
        try:
            run_subject(subj, lut)
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
