# %%
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import h5py
import glob
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

from pathlib import Path

# %%
savedir = Path.cwd().parent.parent / "bin"
print(savedir)
month = "04"
day = "02"
model = "ifs"
if model == "icon":
        model = "icon_d3hp003"
        zoom = 5
elif model == "ifs":
        model = "ifs_tco3999-ng5_rcbmf_cf"
        zoom = 7
path2ds = Path(f"/work/mh0492/m301067/hackaton25/curtains-new/2020/{month}/{day}")
ds = f"ec_curtains_2020{month}{day}_{model}_zoom{zoom}.zarr"
print(path2ds / ds)
ds = xr.open_zarr(path2ds / ds)
ds

# %% ------------------ TRACK COLOR PLOTS ------------------ ###
def plot_track_colormap(ax, cax, data, cmap, norm=None, lab=None, units=True):
        if norm is None:
                norm = Normalize(np.nanmin(data), np.nanmax(data))

        lon, lat = ds.track_lon, ds.track_lat
        ax.scatter(lon, lat, marker=".",
                   s=0.001, c=data, cmap=cmap, norm=norm,
                   transform=ccrs.PlateCarree(central_longitude=180))

        if lab is None:
                lab = f"{data.attrs["name"]}"
                if units:
                        lab +=f" /{data.attrs["units"]}"
        fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, label=lab, extend="both")

def plot_tracks_height_colormap(ax, cax, var, cmap, norm=None, lab=None, units=True):
        bin_edges = np.linspace(-90, 90, 18)
        bin_centers = []
        band_means = []
        for i in range(1, len(bin_edges)):
                l1, l2 = bin_edges[i-1], bin_edges[i]
                bnd = ds.where(ds.track_lat > l1)
                bnd = bnd.where(bnd.track_lat < l2)
                band_means.append(bnd[var].mean(dim="track"))
                bin_centers.append((l1+l2)/2)
        band_means = np.asarray(band_means)

        if norm is None:
                norm = Normalize(np.nanmin(data), np.nanmax(data))

        ax.contourf(ds.level, bin_centers, band_means, cmap=cmap, norm=norm)
        xticks = [1000, 100, 10]
        ax.set_xticks(xticks)

        ax.set_ylabel("Latitude band /degrees")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlabel("Pressure /hPa")
        ax.xaxis.set_label_position("top")

        if lab is None:
                lab = f"{data.attrs["name"]}"
                if units:
                        lab +=f" /{data.attrs["units"]}"
        fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, label=lab, extend="both")


# %%
fig = plt.figure(figsize=(5, 5))
fig.suptitle(f"{model} z{zoom} at surface 2020/{month}{day}")
nrows=2
gs = gridspec.GridSpec(ncols=2, nrows=nrows, figure=fig, width_ratios=[27, 1])

axs, caxs = [], []
for i in range(nrows):
        ax = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
        cax = fig.add_subplot(gs[i, 1])
        ax.set_global()
        ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])
        axs.append(ax)
        caxs.append(cax)

cmap = "inferno"
data = ds.ta.sel(level=ds.level.max())
plot_track_colormap(axs[0], caxs[0], data, cmap)

cmap = "BrBG"
data = ds.hur.sel(level=ds.level.max())
plot_track_colormap(axs[1], caxs[1], data, cmap)

fig.tight_layout()
savename = savedir / f"surfacemap_temp_relh_{model}_zoom{zoom}_2020{month}{day}.png"
plt.savefig(savename, bbox_inches="tight", dpi=180)

# %%
fig = plt.figure(figsize=(5, 8))
fig.suptitle(f"{model} z{zoom} 2020/{month}{day}")
nrows=2
gs = gridspec.GridSpec(ncols=2, nrows=nrows, figure=fig, width_ratios=[27, 1])

axs, caxs = [], []
for i in range(nrows):
        ax = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
        cax = fig.add_subplot(gs[i, 1])
        ax.set_global()
        ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])
        axs.append(ax)
        caxs.append(cax)

cmap = "plasma"
data = ds.prw
norm = LogNorm(1e-3, 10)
plot_track_colormap(axs[0], caxs[0], data, cmap, norm=norm)

cmap = "plasma"
data = ds.clwvi
norm = LogNorm(1e-3, 10)
plot_track_colormap(axs[1], caxs[1], data, cmap, norm=norm)

fig.tight_layout()
savename = savedir / f"surfacemap_columnwater_{model}_zoom{zoom}_2020{month}{day}.png"
plt.savefig(savename, bbox_inches="tight", dpi=180)

# %%
fig = plt.figure(figsize=(5, 10))
fig.suptitle(f"{model} z{zoom} at surface 2020/{month}{day}")
nrows=4
gs = gridspec.GridSpec(ncols=2, nrows=nrows, figure=fig, width_ratios=[27, 1])

axs, caxs = [], []
for i in range(nrows):
        ax = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
        cax = fig.add_subplot(gs[i, 1])
        ax.set_global()
        ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])
        axs.append(ax)
        caxs.append(cax)

norm = LogNorm(1e-9, 1e-3)
cmap = "YlGnBu"
for v, var in enumerate(["clwc", "crwc", "ciwc", "cswc"]):
        data = ds[var].sel(level=ds.level.max())
        plot_track_colormap(axs[v], caxs[v], data, cmap, norm=norm, units=False)

fig.tight_layout()
savename = savedir / f"surfacemap_condensates_{model}_zoom{zoom}_2020{month}{day}.png"
plt.savefig(savename, bbox_inches="tight", dpi=180)

# %%
fig = plt.figure(figsize=(12, 5))
fig.suptitle(f"{model} z{zoom} 2020/{month}{day}")
nrows=2
gs = gridspec.GridSpec(ncols=4, nrows=nrows, figure=fig, width_ratios=[1, 27, 25, 1])

axs, caxs, laxs, lcaxs = [], [], [], []
for i in range(nrows):
        cax = fig.add_subplot(gs[i, 0])
        ax = fig.add_subplot(gs[i, 1], projection=ccrs.Robinson())
        ax.set_global()
        ax.coastlines(color="#333333", linewidth=plt.rcParams["grid.linewidth"])
        axs.append(ax)
        caxs.append(cax)

        laxs.append(fig.add_subplot(gs[i, 2]))
        lcaxs.append(fig.add_subplot(gs[i, 3]))

cmap = "inferno"
var = "ta"
data = ds[var].sel(level=ds.level.max())
norm = Normalize(240, 300)
plot_track_colormap(axs[0], caxs[0], data, cmap, norm=norm)
norm = Normalize(190, 300)
plot_tracks_height_colormap(laxs[0], lcaxs[0], var, cmap, norm=norm)

cmap = "BrBG"
var = "hur"
data = ds[var].sel(level=ds.level.max())
norm = Normalize(10, 125)
plot_track_colormap(axs[1], caxs[1], data, cmap, norm=norm)
plot_tracks_height_colormap(laxs[1], lcaxs[1], var, cmap, norm=norm)

fig.tight_layout()
savename = savedir / f"mapprofile_temp_relh_{model}_zoom{zoom}_2020{month}{day}.png"
plt.savefig(savename, bbox_inches="tight", dpi=180)
