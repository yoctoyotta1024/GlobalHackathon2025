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
ec_day = "2020-05-10"  # format 'YYYY-MM-DD'
ec_time = "T00:00:00"  # format T'hh:mm:ss'
ec_lon_min = 30  # degrees
ec_lon_max = 160  # degrees
ec_lat_min = 6  # degrees
ec_lat_max = 12  # degrees

current_location = "EU"
model = "nicam_gl11"
zoom = 5

output_dir = Path.cwd() / "bin"
### -------------------------------------------- ###


# %% function deinfitions
def interpolate_to_track(ds, track_lon, track_lat, clon_degrees, clat_degrees, day):
    # Define the path for the weights file corresponding to this EC track
    weights_path = Path(
        f"/work/mh0492/m301067/ec-curtains/auxiliary-files/weights/weights_EC_tracks_{day}.nc"
    )

    if weights_path.is_file():
        print("Load existing interpolation weights for this EC track")
        weights = xr.open_dataset(weights_path)
    else:
        print("Compute weights using Delaunay triangulation")
        weights = egr.compute_weights_delaunay(
            points=(clon_degrees, clat_degrees),
            xi=(track_lon, track_lat),
        )
        # Save weights to a NetCDF file for future use
        weights.to_netcdf(weights_path)

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
ec_datetime = np.datetime64(f"{ec_day}{ec_time}")
start_time = ec_datetime - np.timedelta64(30, "m")
end_time = ec_datetime + np.timedelta64(30, "m")

ec_track = get_earthcare_track(ec_day, start_time, end_time)  # TODO(ALL): get EC track
ec_track_lon = ec_track.lon
ec_track_lat = ec_track.lat
ec_track_time = ec_track.time

# %% Trim track coordinates to be within the lat/lon bounds
valid_indices = np.where(
    (ec_track_lon >= ec_lon_min)
    & (ec_track_lon <= ec_lon_max)
    & (ec_track_lat >= ec_lat_min)
    & (ec_track_lat <= ec_lat_min)
)[0]
ec_track_lon = ec_track_lon[valid_indices]
ec_track_lat = ec_track_lat[valid_indices]
ec_track_time = ec_track_time[valid_indices]

# %% Interpolating the dataset to the EarthCARE track
ds_lon_deg = np.degrees(ds["clon"].values)
ds_lat_deg = np.degrees(ds["clat"].values)
ds_curtain = interpolate_to_track(
    ds, ec_track_lon, ec_track_lat, ds_lon_deg, ds_lat_deg, ec_day
)

# Add track longitude, latitude, and time to the curtain datafile
ds_curtain = ds_curtain.assign(
    track_lon=("track", ec_track_lon.data),
    track_lat=("track", ec_track_lat.data),
    track_time=("track", ec_track_time.data),
)

# %% save curtain data to netcdf file
output_dir.mkdir(exist_ok=True)
print(f"Writing curtain profiles in {output_dir}")
savelabel = ec_day + ec_time.replace(":", "-")
curtain_file = output_dir / f"orcestra_ec-curtain_{savelabel}.nc"
ds_curtain.to_netcdf(curtain_file)
print(f"Curtain extracted and saved in {curtain_file}")
