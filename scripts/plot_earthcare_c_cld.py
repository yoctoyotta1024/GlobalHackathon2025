# test_plot_earthcare.py

import os
import glob
import numpy as np
import numpy.ma as ma
import h5py
import xarray as xr
from datetime import datetime, timedelta
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from itertools import groupby
from operator import itemgetter
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ===================================================
# Constants and File Paths
# ===================================================
REF_FILE_PATH = "/work/bb1153/b383455/EarthCARE_CPR/C-PRO/C-FMR/2025/04"
DOPPLER_FILE_PATH = "/work/bb1153/b383455/EarthCARE_CPR/C-PRO/C-CD/2025/04"
CLD_FILE_PATH = "/work/bb1153/b383455/EarthCARE_CPR/C-CLD/2025/04"
INVALID_VALUE = 9.96920997e+36
INVALID_THRESHOLD = 1e+30

# choose one orbit for test (e.g. '04951')
ORBIT_ID = "04951D"

# ===================================================
# Utility Functions
# ===================================================
def list_cld_variables_and_units(cld_file_path):
    """
    Print all variable names and units from a C-CLD file.
    """
    print(f"\nScanning CLD file: {cld_file_path}")
    try:
        with h5py.File(cld_file_path, "r") as f:
            def print_attrs(name, obj):
                if isinstance(obj, h5py.Dataset):
                    unit = obj.attrs.get("units", "(no unit)")
                    if isinstance(unit, bytes):
                        unit = unit.decode()
                    print(f"{name:50} — {unit}")
            f.visititems(print_attrs)
    except Exception as e:
        print(f"Failed to read file: {e}")

def get_file_dict_by_orbit(path):
    """
    Returns a dict mapping orbit+frame token (e.g., '04451D') to file path.
    """
    files = sorted(glob.glob(os.path.join(path, "*.h5")))
    file_dict = {}
    for fpath in files:
        token = os.path.basename(fpath).split('_')[-1].split('.')[0]
        file_dict[token] = fpath
    return file_dict

def find_fmr_files(orbit_id, ref_dir):
    """Return sorted list of FMR files containing orbit_id in their name."""
    pattern = os.path.join(ref_dir, f"*{orbit_id}*.h5")
    files = sorted(glob.glob(pattern))
    return files

def find_dop_files(orbit_id, dop_dir):
    """Return sorted list of DOP files containing orbit_id in their name."""
    pattern = os.path.join(dop_dir, f"*{orbit_id}*.h5")
    files = sorted(glob.glob(pattern))
    return files

def find_cld_files(orbit_id, cld_dir):
    """Return sorted list of CLD files containing orbit_id in their name."""
    pattern = os.path.join(cld_dir, f"*{orbit_id}*.h5")
    files = sorted(glob.glob(pattern))
    return files

def compute_alongtrack(lat, lon):
    """Dummy along-track distance function (real one uses haversine)."""
    return np.arange(len(lat)) * 1.0  # placeholder: 1 km spacing

def load_lajolla_cmap(filepath):
    data = np.loadtxt(filepath)
    return LinearSegmentedColormap.from_list('lajolla', data)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    return LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n)))

def stitch_c_pro(ref_paths, dop_paths, cld_paths):
    """Read reflectivity and Doppler velocity arrays from one FMR, CD, and CLD files."""
    refl_list = []
    dop_list = []
    lat_list = []
    lon_list = []
    time_list = []
    height_list = []
    selev_list = []
    land_list = []
    pia_list = []
    sed_list = []
    lwc_list = []
    lreff_list = []
    wc_list = []
    lwp_list = []
    iwp_list = []
    rwp_list = []
    class_list = []
    frame_list = []
    
    for ref_path, dop_path, cld_path in zip(ref_paths, dop_paths, cld_paths):
        with h5py.File(ref_path, 'r') as rf, h5py.File(dop_path, 'r') as df, h5py.File(cld_path, 'r') as cf:
            # Groups
            ref_grp = rf["ScienceData"]
            dop_grp = df["ScienceData"]
            cld_grp = cf["ScienceData"]
            # Coordinates
            lat = ref_grp['latitude'][:]
            lon = ref_grp['longitude'][:]
            time_sec = ref_grp['time'][:]
            height = ref_grp['height'][:]
            height = np.where(height > INVALID_THRESHOLD, 0.0, height) / 1000.0
            # Convert to datetime
            time_arr = np.array([
                datetime(2000,1,1) + timedelta(seconds=float(t))
                for t in time_sec
            ])

            refl = ref_grp['reflectivity_corrected'][:]
            selev = ref_grp['surface_elevation'][:]
            land = ref_grp['land_flag'][:]
            pia = ref_grp['path_integrated_attenuation'][:]
            dop = dop_grp['doppler_velocity_best_estimate'][:]
            sed = dop_grp['sedimentation_velocity_best_estimate'][:]
            lwc = cld_grp['liquid_water_content'][:]
            lreff = cld_grp['liquid_effective_radius'][:]
            wc = cld_grp['water_content'][:]
            lwp = cld_grp['liquid_water_path'][:]
            iwp = cld_grp['ice_water_path'][:]
            rwp = cld_grp['rain_water_path'][:]
            hydclass = cld_grp['hydrometeor_classification'][:]

            fname = os.path.basename(ref_path)
            parts = fname.split("_")
            frame_code = parts[7][5]  # e.g., '04951A' → 'A'
            frame_arr = np.array([frame_code] * refl.shape[0])

            # append everything
            refl_list.append(refl)
            dop_list.append(dop)
            lat_list.append(lat)
            lon_list.append(lon)
            time_list.append(np.array(time_arr, dtype="datetime64[ns]"))
            height_list.append(height)
            selev_list.append(selev)
            land_list.append(land)
            pia_list.append(pia)
            sed_list.append(sed)
            lwc_list.append(lwc)
            lreff_list.append(lreff)
            wc_list.append(wc)
            lwp_list.append(lwp)
            iwp_list.append(iwp)
            rwp_list.append(rwp)
            class_list.append(hydclass)
            frame_list.append(frame_arr)

    return {
        "latitude": np.concatenate(lat_list),
        "longitude": np.concatenate(lon_list),
        "time": np.concatenate(time_list),
        "height":  np.concatenate(height_list),
        "reflectivity_corrected": np.vstack(refl_list),
        "doppler_velocity_best_estimate": np.vstack(dop_list),
        "sedimentation_velocity_best_estimate": np.vstack(sed_list),
        "surface_elevation": np.concatenate(selev_list),
        "land_flag": np.concatenate(land_list),
        "path_integrated_attenuation": np.concatenate(pia_list),
        "liquid_water_content": np.concatenate(lwc_list),
        "liquid_effective_radius": np.concatenate(lreff_list),
        "water_content": np.concatenate(wc_list),
        "liquid_water_path": np.concatenate(lwp_list),
        "ice_water_path": np.concatenate(iwp_list),
        "rain_water_path": np.concatenate(rwp_list),
        "hydrometeor_classification": np.concatenate(class_list),
        "frame": np.concatenate(frame_list),
    }

def plot_reflectivity_curtain_from_result(result, lajolla_txt_path, title="Reflectivity Curtain"):
    refl = result["reflectivity_corrected"]
    dop = result["doppler_velocity_best_estimate"]
    height = result["height"]
    lat = result["latitude"]
    lon = result["longitude"]
    selev = result["surface_elevation"] / 1000.0  # convert to km
    frame = result["frame"]  # e.g., array like ['A', 'A', 'A', ..., 'B', 'B', ...]
    
    refl_plot = refl.astype(float)
    mask_invalid = refl_plot > INVALID_THRESHOLD
    mask_weak    = refl_plot < -35
    refl_plot[mask_invalid | mask_weak] = np.nan

    dop_plot = dop.astype(float)
    mask_invalid = dop_plot > INVALID_THRESHOLD
    mask_ref     = refl_plot < -21
    dop_plot[mask_invalid | mask_ref | mask_weak] = np.nan

    # Along-track distance
    x_km = compute_alongtrack(lat, lon)
    x_mesh = np.tile(x_km[:, np.newaxis], (1, height.shape[1]))

    # Colormap
    base_cmap = load_lajolla_cmap(lajolla_txt_path)
    lajolla_cmap = truncate_colormap(base_cmap, 0.1, 0.95).reversed()

    # Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 2], hspace=0.25)

    # Top Panel: Map with track colored by frame
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[1, 0.2, 1], wspace=0.01)
    ax0 = fig.add_subplot(gs_top[0], projection=ccrs.PlateCarree())
    ax0.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax0.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax0.set_title("EarthCARE Track", fontsize=11)

    ax0.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
    ax0.set_yticks([-80, -60, -30, 0, 30, 60, 80], crs=ccrs.PlateCarree())
    ax0.tick_params(labelsize=8)
    #ax0.set_xlabel("Longitude", fontsize=9)
    ax0.set_ylabel("Latitude", fontsize=9)

    unique_frames = sorted(set(frame))
    color_map = {f: plt.cm.tab10(i % 10) for i, f in enumerate(unique_frames)}
    MAX_LON_JUMP = 180

    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]

        first = True
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            lon_group = lon[group]
            lat_group = lat[group]
            jump_indices = np.where(np.abs(np.diff(lon_group)) > MAX_LON_JUMP)[0]

            if len(jump_indices) == 0:
                ax0.plot(lon_group, lat_group, '-', label=f"Frame {f}" if first else None, color=color_map[f], linewidth=2.5)
                first = False
            else:
                split_points = np.concatenate(([0], jump_indices + 1, [len(lon_group)]))
                for i in range(len(split_points) - 1):
                    i1, i2 = split_points[i], split_points[i+1]
                    if i2 - i1 >= 2:
                        ax0.plot(lon_group[i1:i2], lat_group[i1:i2], '-', label=f"Frame {f}" if first else None, color=color_map[f], linewidth=2.5)
                        first = False

    ax0.set_extent([np.min(lon) - 1, np.max(lon) + 1, np.min(lat) - 1, np.max(lat) + 1])
    for spine in ax0.spines.values():
        spine.set_linewidth(1.5)

    ax_legend = fig.add_subplot(gs_top[1])
    ax_legend.axis('off')

    handles = [plt.Line2D([0], [0], color=color_map[f], lw=2) for f in unique_frames]
    labels = [f"{f}" for f in unique_frames]
    ax_legend.legend(
        handles, labels,
        loc='upper left',
        fontsize=9,
        title="Frame",
        title_fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.05, 1.0), 
        borderaxespad=0.0
    )
    ax_legend.legend(handles, labels, loc='center', fontsize=9, title="Frame", title_fontsize=10, frameon=False)

    ax_text = fig.add_subplot(gs_top[2])
    ax_text.axis('off')
    frame_range = f"{min(frame)}–{max(frame)}"
    #orbit_text = f"{ORBIT_ID}{frame_range}"  # e.g., 04151A–D
    orbit_text = f"{ORBIT_ID}"  # e.g., 04151D
    ax_text.text(0.0, 0.6, orbit_text, fontsize=13, va='bottom', ha='left', transform=ax_text.transAxes)
    t_start = np.min(result["time"]).astype('M8[ms]').astype(datetime)
    t_end   = np.max(result["time"]).astype('M8[ms]').astype(datetime)
    period_str = f"{t_start:%Y.%m.%d %H:%M:%S} – {t_end:%m.%d %H:%M}"
    ax_text.text(0.0, 0.4, period_str, fontsize=13, va='center', ha='left', transform=ax_text.transAxes)

    # Middle Panel: Reflectivity
    ax1 = fig.add_subplot(gs[1])
    pcm1 = ax1.pcolormesh(x_mesh, height, refl_plot, cmap=lajolla_cmap, shading="auto", vmin=-35, vmax=20)
    for i in range(len(x_km) - 1):
        ax1.fill([x_km[i], x_km[i+1], x_km[i+1], x_km[i]], [0, 0, selev[i+1], selev[i]], color="lightgray", alpha=0.5)
    ax1.set_ylim(0, 18)
    ax1.set_ylabel("Height (km)")
    ax1.set_title("Corrected Reflectivity (C-FMR)", fontsize=11)
    ax1.grid(True, linestyle='-', linewidth=0.35, alpha=0.4)
    cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.01, aspect=15)
    cbar1.set_label("Z (dBZ)")
    y_bar = 17.5
    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            x1 = x_km[group[0]]
            x2 = x_km[group[-1]]
            ax1.fill_between([x1, x2], y_bar - 0.15, y_bar + 0.15, color=color_map[f], alpha=0.9, linewidth=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Bottom Panel: Doppler velocity
    ax2 = fig.add_subplot(gs[2])
    bounds = np.arange(-1.7, 1.9, 0.2)
    colors = ['#d62747', '#f54a6c', '#e2617b', '#a3745d', '#c1884f',
              '#e3a43f', '#ffc130', '#f2e238', '#e8e7e9', '#9ed2a4',
              '#6fc673', '#25a630', '#178d1f', '#70b4d3', '#785ee0',
              '#501f9f', '#88108c']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    pcm2 = ax2.pcolormesh(x_mesh, height, dop_plot, cmap=cmap, norm=norm, shading="auto")
    for i in range(len(x_km) - 1):
        ax2.fill([x_km[i], x_km[i+1], x_km[i+1], x_km[i]], [0, 0, selev[i+1], selev[i]], color="lightgray", alpha=0.5)
    ax2.set_ylim(0, 18)
    ax2.set_ylabel("Height (km)")
    ax2.set_xlabel("Along-track Distance (km)")
    ax2.set_title("Doppler Velocity Best Estimate (C-CD)", fontsize=11)
    ax2.grid(True, linestyle='-', linewidth=0.35, alpha=0.4)
    cbar2 = plt.colorbar(pcm2, ax=ax2, pad=0.01, aspect=15)
    cbar2.set_label("Velocity (m/s)")
    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            x1 = x_km[group[0]]
            x2 = x_km[group[-1]]
            ax2.fill_between([x1, x2], y_bar - 0.15, y_bar + 0.15, color=color_map[f], alpha=0.9, linewidth=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    return fig, (ax0, ax1, ax2)

def plot_cld_curtain_from_result(result):
    refl = result["reflectivity_corrected"]
    lwc = result["liquid_water_content"] # kg/m3
    lreff = result["liquid_effective_radius"] # m
    lwp = result["liquid_water_path"] # kg/m2
    wc = result["water_content"] # kg/m3
    rwp = result["rain_water_path"] # kg/m2
    iwp = result["ice_water_path"] # kg/m2
    hydclass = result["hydrometeor_classification"]
    height = result["height"]
    lat = result["latitude"]
    lon = result["longitude"]
    selev = result["surface_elevation"] / 1000.0  # convert to km
    frame = result["frame"]  # e.g., array like ['A', 'A', 'A', ..., 'B', 'B', ...]

    wc_plot = wc.astype(float)
    refl_plot = refl.astype(float)
    mask_invalid = wc_plot > INVALID_THRESHOLD
    mask_weak    = refl_plot < -35
    wc_plot[mask_invalid | mask_weak] = np.nan

    hydclass_plot = hydclass.astype(float)
    mask_invalid = hydclass_plot > INVALID_THRESHOLD
    hydclass_plot[mask_invalid] = np.nan

    # Along-track distance
    x_km = compute_alongtrack(lat, lon)
    x_mesh = np.tile(x_km[:, np.newaxis], (1, height.shape[1]))

    # Plot
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 2], hspace=0.25)

    # Top Panel: Map with track colored by frame
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[1, 0.2, 1], wspace=0.01)
    ax0 = fig.add_subplot(gs_top[0], projection=ccrs.PlateCarree())
    ax0.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
    ax0.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax0.set_title("EarthCARE Track", fontsize=11)

    ax0.set_xticks([-180, -90, 0, 90, 180], crs=ccrs.PlateCarree())
    ax0.set_yticks([-80, -60, -30, 0, 30, 60, 80], crs=ccrs.PlateCarree())
    ax0.tick_params(labelsize=8)
    #ax0.set_xlabel("Longitude", fontsize=9)
    ax0.set_ylabel("Latitude", fontsize=9)

    unique_frames = sorted(set(frame))
    color_map = {f: plt.cm.tab10(i % 10) for i, f in enumerate(unique_frames)}
    MAX_LON_JUMP = 180

    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]

        first = True
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            lon_group = lon[group]
            lat_group = lat[group]
            jump_indices = np.where(np.abs(np.diff(lon_group)) > MAX_LON_JUMP)[0]

            if len(jump_indices) == 0:
                ax0.plot(lon_group, lat_group, '-', label=f"Frame {f}" if first else None, color=color_map[f], linewidth=2.5)
                first = False
            else:
                split_points = np.concatenate(([0], jump_indices + 1, [len(lon_group)]))
                for i in range(len(split_points) - 1):
                    i1, i2 = split_points[i], split_points[i+1]
                    if i2 - i1 >= 2:
                        ax0.plot(lon_group[i1:i2], lat_group[i1:i2], '-', label=f"Frame {f}" if first else None, color=color_map[f], linewidth=2.5)
                        first = False

    ax0.set_extent([np.min(lon) - 1, np.max(lon) + 1, np.min(lat) - 1, np.max(lat) + 1])
    for spine in ax0.spines.values():
        spine.set_linewidth(1.5)

    ax_legend = fig.add_subplot(gs_top[1])
    ax_legend.axis('off')

    handles = [plt.Line2D([0], [0], color=color_map[f], lw=2) for f in unique_frames]
    labels = [f"{f}" for f in unique_frames]
    ax_legend.legend(
        handles, labels,
        loc='upper left',
        fontsize=9,
        title="Frame",
        title_fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.05, 1.0), 
        borderaxespad=0.0
    )
    ax_legend.legend(handles, labels, loc='center', fontsize=9, title="Frame", title_fontsize=10, frameon=False)

    ax_text = fig.add_subplot(gs_top[2])
    ax_text.axis('off')
    frame_range = f"{min(frame)}–{max(frame)}"
    orbit_text = f"{ORBIT_ID}{frame_range}"  # e.g., 04151A–D
    ax_text.text(0.0, 0.6, orbit_text, fontsize=13, va='bottom', ha='left', transform=ax_text.transAxes)
    t_start = np.min(result["time"]).astype('M8[ms]').astype(datetime)
    t_end   = np.max(result["time"]).astype('M8[ms]').astype(datetime)
    period_str = f"{t_start:%Y.%m.%d %H:%M:%S} – {t_end:%m.%d %H:%M}"
    ax_text.text(0.0, 0.4, period_str, fontsize=13, va='center', ha='left', transform=ax_text.transAxes)

    # Middle Panel: Water content
    ax1 = fig.add_subplot(gs[1])
    pcm1 = ax1.pcolormesh(x_mesh, height, wc_plot, cmap="YlGnBu", shading="auto", vmin=1e-7, vmax=1e-4)
    for i in range(len(x_km) - 1):
        ax1.fill([x_km[i], x_km[i+1], x_km[i+1], x_km[i]], [0, 0, selev[i+1], selev[i]], color="lightgray", alpha=0.5)
    ax1.set_ylim(0, 18)
    ax1.set_ylabel("Height (km)")
    ax1.set_title("Water Content (ice, snow, and rain)", fontsize=11)
    ax1.grid(True, linestyle='-', linewidth=0.35, alpha=0.4)
    cbar1 = plt.colorbar(pcm1, ax=ax1, pad=0.01, aspect=15)
    cbar1.set_label("(kg/m3)")
    y_bar = 17.5
    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            x1 = x_km[group[0]]
            x2 = x_km[group[-1]]
            ax1.fill_between([x1, x2], y_bar - 0.15, y_bar + 0.15, color=color_map[f], alpha=0.9, linewidth=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)

    # Bottom Panel: Hydrometeor Classification
    ax2 = fig.add_subplot(gs[2])
    # Set the classification range and corresponding colors (-1 to 20)
    bounds = np.arange(-1, 22)
    colors = ['#000000', '#a26640', '#ffffff', '#ffff8b', '#f5be26',
            '#f97415', '#ff000c', '#bf71fb', '#004576', '#0064f9',
            '#95d1fb', '#d7fffe', '#7b971d', '#840000', '#0205a7',
            '#840000', '#001145', '#bb3f3f', '#5684ad', '#eedc64',
            '#d8dcd6', '#c5c9c7']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    pcm2 = ax2.pcolormesh(x_mesh, height, hydclass_plot, cmap=cmap, norm=norm, shading="auto")
    for i in range(len(x_km) - 1):
        ax2.fill([x_km[i], x_km[i+1], x_km[i+1], x_km[i]], [0, 0, selev[i+1], selev[i]], color="lightgray", alpha=0.5)
    ax2.set_ylim(0, 18)
    ax2.set_ylabel("Height (km)")
    ax2.set_xlabel("Along-track Distance (km)")
    ax2.set_title("Hydrometeor Classification", fontsize=11)
    ax2.grid(True, linestyle='-', linewidth=0.35, alpha=0.4)
    tick_positions = (bounds[:-1] + bounds[1:]) / 2
    cbar = plt.colorbar(pcm2, ax=ax2, orientation='vertical', pad=0.02, aspect=15)
    cbar.set_ticks(tick_positions)
    TC_name = ['missing data','sub-surface','clear','liquid cloud','drizzling liquid cloud',
                'warm rain','cold rain','melting snow','rimed snow','snow','ice cloud','stratospheric ice',
                'insects','heavy rain likely','heavy snow likely','heavy rain','heavy snow','rain in clutter',
                'snow in clutter','cloud in clutter','clear in clutter','uncertain']
    cbar.set_ticklabels([str(i) for i in TC_name[:]])
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())
    for f in unique_frames:
        mask = (frame == f)
        indices = np.where(mask)[0]
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            x1 = x_km[group[0]]
            x2 = x_km[group[-1]]
            ax2.fill_between([x1, x2], y_bar - 0.15, y_bar + 0.15, color=color_map[f], alpha=0.9, linewidth=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    return fig, (ax0, ax1, ax2)

# ===================================================
# Main Processing Function
# ===================================================
def main():
    print("Scanning FMR, CD, and CLD directories...", flush=True)
    ref_dict = get_file_dict_by_orbit(REF_FILE_PATH)
    dop_dict = get_file_dict_by_orbit(DOPPLER_FILE_PATH)
    cld_dict = get_file_dict_by_orbit(CLD_FILE_PATH)
    keys = sorted(set(ref_dict.keys()) & set(dop_dict.keys()) & set(cld_dict.keys()))
    print(f"Found {len(keys)} matching orbit files.", flush=True)
    if not keys:
        print("No matching files to process.", flush=True)
        return

    fmr_files = find_fmr_files(ORBIT_ID, REF_FILE_PATH)
    if not fmr_files:
        print(f"No FMR files found for orbit {ORBIT_ID}")
        return
    dop_files = find_dop_files(ORBIT_ID, DOPPLER_FILE_PATH)
    if not dop_files:
        print(f"No DOP files found for orbit {ORBIT_ID}")
        return
    cld_files = find_cld_files(ORBIT_ID, CLD_FILE_PATH)
    if not cld_files:
        print(f"No CLD files found for orbit {ORBIT_ID}")

    #sample_cld_path = cld_dict[keys[0]]
    #list_cld_variables_and_units(sample_cld_path)

    data_lists = {k: [] for k in [
        "latitude","longitude","time",
        "reflectivity_corrected",
        "height","surface_elevation","land_flag",
        "path_integrated_attenuation",
        "doppler_velocity_best_estimate","sedimentation_velocity_best_estimate",
        "liquid_water_content", "liquid_effective_radius", "liquid_water_path",
        "water_content",
        "rain_water_path", "ice_water_path",
        "hydrometeor_classification"
    ]}
    result = stitch_c_pro(fmr_files, dop_files, cld_files)

    #lajolla_path="/home/b/b383455/colormaps/lajolla/lajolla.txt"

    # FMR and CD
    #fig, ax = plot_reflectivity_curtain_from_result(result, lajolla_path) 
    #outname = f"EarthCARE_Curtains.png"
    #fig.savefig(outname, dpi=300, bbox_inches='tight')
    #plt.close(fig)
    #print(f"  -> Saved: {outname}")

    # CLD
    fig_cld, ax = plot_cld_curtain_from_result(result)
    outname = f"EarthCARE_CLD_Curtains.png"
    fig_cld.savefig(outname, dpi=300, bbox_inches='tight')
    plt.close(fig_cld)
    print(f"  -> Saved: {outname}")

if __name__ == "__main__":
    main()