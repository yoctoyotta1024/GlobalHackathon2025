# %% Import modules
import easygems.remap as egr
from easygems import healpix as egh
import intake
import numpy as np
import xarray as xr
import warnings

from pathlib import Path

from data_utils import datetime_from_data_array_time

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts

# %% ------------- INPUT PARAMETERS ------------- ###
# Earthcare file to get lat-lon track from
ec_year = "2024" # format 'YYYY'
ec_month = "08" # format 'MM'
ec_day = "01" # format 'DD'
ec_start_time = "T0019"  # format T'hhmm'
ec_end_time = "T0031"  # format T'hhmm'

ec_start_date = ec_year + ec_month + ec_day + ec_start_time
ec_end_date = ec_year + ec_month + ec_day + ec_end_time
ec_data_path = Path("/work") / "mh0731" / "m301196" / "ecomip"/ "ftp.eorc.jaxa.jp" / "eorc" / "CPR" / "1B" / "xCa" / ec_year / ec_month / ec_day
ec_file = ec_data_path / f"ECA_J_CPR_NOM_1BS_{ec_start_date}_{ec_end_date}_00997A_vCa_corr_xCa.nc"

# (optional) limits to lat-lon range of earthcare path
ec_lon_min = None # degrees
ec_lon_max = None # degrees
ec_lat_min = None # degrees
ec_lat_max = None # degrees

# model, zoom level, and time for curtain dataset
current_location = "EU"
model = "icon_d3hp003"
model = "icon_art_lam"
zoom = 10
model_year = "2020" # format 'YYYY'
model_month = ec_month # format 'MM'
model_day = ec_day # format 'DD'
model_time = ec_start_time  # format T'hhmm'

# name of exisiting or to-be-created weights file for this ec track and model
# weights_dir = Path("/work") / "mh0492" / "m301067" / "hackaton25" / "auxiliary-files" / "weights"
weights_dir = Path.cwd() / "weights"
weights_label = f"{ec_start_date}_{ec_end_date}_{model}_zoom{zoom}"
weights_file = weights_dir / f"weights_ec_tracks_{weights_label}.nc"

# name of to-be-created curtain .zarr dataset for this ec track and model
curtain_dir = Path("/work") / "mh0492" / "m301067" / "hackaton25" / "curtains" / model_year / model_month / model_day
curtain_dir = Path.cwd() / "curtains"
curtain_label = f"{ec_start_date}_{ec_end_date}_{model}_zoom{zoom}_aero"
curtain_file = curtain_dir / f"ec_curtain_{curtain_label}.zarr"
### -------------------------------------------- ###

# %% function definitions
def read_earthcare_track(ec_file, engine_type="netcdf4"):
    var = xr.open_dataset(ec_file, engine=engine_type)

    lon = np.array(var['lon']).astype(np.float64)
    lat = np.array(var['lat']).astype(np.float64)

    return lon, lat

def interpolate_to_track(ds, weights_file, track_lon, track_lat=None):
    if weights_file.is_file():
        print("loading existing interpolation weights for this EC track")
        weights = xr.open_dataset(weights_file)
    else:
        print("computing weights using Delaunay triangulation")
        ds = (
            ds.rename_dims({"value": "cell"}).pipe(egh.attach_coords)
            if "value" in ds.dims
            else ds.pipe(egh.attach_coords)
        )
        weights = egr.compute_weights_delaunay(
            points=(ds["lon"].values, ds["lat"].values),
            xi=(track_lon, track_lat),
        )
        weights.to_netcdf(weights_file)

    # Apply weights to interpolate the dataset
    ds_interpolated = xr.apply_ufunc(
        egr.apply_weights,
        ds,
        kwargs=weights,
        input_core_dims=[["cell"]],
        output_core_dims=[["track"]],
        output_dtypes=["f4"],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={
            "output_sizes": {"track": len(track_lon)},
        },
        keep_attrs=True,
    )

    return ds_interpolated

# %% Load catalog and dataset at time nearest model time given
# cat = intake.open_catalog(
#     "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
# )[current_location]

model_hour = model_time[1:3]
model_min = model_time[3:6]
model_datetime = np.datetime64(f"{model_year}-{model_month}-{model_day}T{model_hour}:{model_min}")
# ds = cat[model](zoom=zoom).to_dask().sel(time=model_datetime, method="nearest")

store = 'aerosols.zarr'
data_path = Path('/work/bb1215/b382013/output/processed/Africa-dust-lam/20231222/default/', store)
ds = xr.open_dataset(data_path)
ds = ds.assign_coords(dates=("time", datetime_from_data_array_time(ds.time)))
ds = ds.sel(time=model_datetime, method="nearest")

# %% Load the EarthCARE track and select the time range
ec_track_lon, ec_track_lat = read_earthcare_track(ec_file)

# %% Trim track coordinates to be within the lat/lon bounds
if ec_lon_min is not None and ec_lon_max is not None and ec_lat_min is not None and ec_lat_max is not None:
    valid_indices = np.where(
        (ec_track_lon >= ec_lon_min)
        & (ec_track_lon <= ec_lon_max)
        & (ec_track_lat >= ec_lat_min)
        & (ec_track_lat <= ec_lat_min)
    )[0]
    ec_track_lon = ec_track_lon[valid_indices]
    ec_track_lat = ec_track_lat[valid_indices]

# %% Interpolating the dataset to the EarthCARE track
ds_curtain = interpolate_to_track(
    ds, weights_file, ec_track_lon, track_lat=ec_track_lat,
)

# Add track longitude, latitude, and time to the curtain datafile
ds_curtain = ds_curtain.assign(
    track_lon=("track", ec_track_lon.data),
    track_lat=("track", ec_track_lat.data),
)
ds_curtain.attrs.update({
    "model_datetime": str(model_datetime),
    "ec_track_start_date": ec_start_date,
    "ec_track_end_date": ec_end_date,
    "ec_track_date_format": 'YYYYMMDDThhmm'
})
# %% save curtain data to netcdf file
curtain_dir.mkdir(parents=True, exist_ok=True)
print(f"Writing curtain profiles in '{curtain_file}'")
ds_curtain.to_zarr(curtain_file)
print(f"Curtain extracted and saved in {curtain_file.name}")
