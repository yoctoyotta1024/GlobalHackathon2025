# %% Import modules
import easygems.remap as egr
import intake
import numpy as np
import xarray as xr
import warnings

from pathlib import Path

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # don't warn us about future package conflicts

# %% ------------- INPUT PARAMETERS ------------- ###
ec_year = "2024" # format 'YYYY'
ec_month = "08" # format 'MM'
ec_day = "01" # format 'DD'
ec_start_time = "T0008"  # format T'hhmm'
ec_end_time = "T0019"  # format T'hhmm'
ec_lon_min = None # degrees
ec_lon_max = None # degrees
ec_lat_min = None # degrees
ec_lat_max = None # degrees

ec_start_date = ec_year + ec_month + ec_day + ec_start_time
ec_end_date = ec_year + ec_month + ec_day + ec_end_time
ec_data_path = Path("/work") / "mh0731" / "m301196" / "ecomip"/ "ftp.eorc.jaxa.jp" / "eorc" / "CPR" / "1B" / "xCa" / ec_year / ec_month / ec_day
ec_file = ec_data_path / f"ECA_J_CPR_NOM_1BS_{ec_start_date}_{ec_end_date}_00996H_vCa_corr_xCa.nc"

current_location = "EU"
model = "icon_d3hp003"
zoom = 5

output_dir = Path.cwd() / "bin"

# define name of exisiting or to-be-created weights file for this EC track
weights_label =  f"{ec_start_date}_{ec_end_date}"
weights_file = Path(
    "/work") / "mh0492" / "m301067" / "hackaton25" / "auxiliary-files" / "weights" / "weights_ec_tracks_{weights_label}.nc"
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
        ds_lon_deg = np.degrees(ds["clon"].values)
        ds_lat_deg = np.degrees(ds["clat"].values)
        weights = egr.compute_weights_delaunay(
            points=(ds_lon_deg, ds_lat_deg),
            xi=(track_lon, track_lat),
        )
        weights.to_netcdf(weights_file)

    # Apply weights to interpolate the dataset
    ds_interpolated = xr.apply_ufunc(
        egr.apply_weights,
        ds,
        kwargs=weights,
        input_core_dims=[["ncells"]],
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

# %% Load catalog and dataset
cat = intake.open_catalog(
    "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
)[current_location]
ds = cat[model](zoom=zoom).to_dask()

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
dataset.assign_attrs(
    start_date=ec_start_date,
    end_date=ec_end_date,
    date_format='YYYYMMDDThhmm'
)

# %% save curtain data to netcdf file
output_dir.mkdir(exist_ok=True)
print(f"Writing curtain profiles in {output_dir}")
curtain_label = f"{ec_start_date}_{ec_end_date}_{model}_zoom{zoom}"
curtain_file = output_dir / f"orcestra_ec-curtain_{curtain_label}.nc"
ds_curtain.to_netcdf(curtain_file)
print(f"Curtain extracted and saved in {curtain_file}")
