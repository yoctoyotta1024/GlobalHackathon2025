import argparse
import easygems.remap as egr
import glob
import h5py
import intake
import numpy as np
import shutil
import os
import warnings
import xarray as xr

from easygems import healpix as egh
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

def read_earthcare_track(ec_file, engine_type="h5"):
    if engine_type == "h5":
        lon = h5py.File(ec_file, "r")["ScienceData/Geo/longitude"][:]
        lat = h5py.File(ec_file, "r")["ScienceData/Geo/latitude"][:]
    elif engine_type == "netcdf4":
        var = xr.open_dataset(ec_file, engine=engine_type)
        lon = np.array(var['lon']).astype(np.float64)
        lat = np.array(var['lat']).astype(np.float64)
    else:
        raise ValueError(f"engine type {engine_type} not supported")
    return lon, lat

def interpolate_to_track(ds, weights_file, track_lon, track_lat=None):
    if weights_file.is_file():
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
        os.chmod(weights_file, 0o777)

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
            "allow_rechunk": True,
        },
        keep_attrs=True,
    )

    return ds_interpolated

def coarsen_dataset(ds, nlevels_coarsen):
  # each coarsening by 4 will reduce the data by one zoom level
  return ds.coarsen(cell=4**nlevels_coarsen).mean()

def create_curtains_dataset(model, zoom, ds, ec_files, weights_dir):
    weights_dir.mkdir(parents=True, exist_ok=True)

    curtains = []
    for f in ec_files:
        # Load the EarthCARE track and select the time range
        ec_track_lon, ec_track_lat = read_earthcare_track(f, engine_type="h5")

        # Interpolating the dataset to the EarthCARE track
        weights_label = f"{Path(f).stem}_{model}_zoom{zoom}"
        weights_file = weights_dir / f"weights_ec_tracks_{weights_label}.nc"
        ds_curtain = interpolate_to_track(
            ds, weights_file, ec_track_lon, track_lat=ec_track_lat,
        )

        # Add track longitude, latitude, and time to the curtain datafile
        ds_curtain = ds_curtain.assign(
            track_lon=("track", ec_track_lon.data),
            track_lat=("track", ec_track_lat.data),
        )
        curtains.append(ds_curtain)
    curtains = xr.concat(curtains, dim="track")

    return curtains

def write_curtain(model, zoom, date, current_location="EU", nlevels_coarsen=0):
    ec_year, ec_month, ec_day = date.split('/')
    ec_data_path = Path("/work") / "mh0731" / "m301196" / "ecomip" / "ftp.eorc.jaxa.jp" / "CPR" / "2A" / "CPR_CLP" / "vBa" / ec_year / ec_month / ec_day
    ec_file_search = f"ECA_J_CPR_CLP_2AS_{ec_year}{ec_month}{ec_day}T*_vBa.h5"
    ec_files = sorted(glob.glob(str(ec_data_path / ec_file_search)))

    model_year = "2020"
    model_month = ec_month
    model_day = ec_day

    weights_dir = Path("/work") / "mh0492" / "m301067" / "hackaton25" / "auxiliary-files" / "weights" / ec_year / ec_month / ec_day
    curtain_dir = Path("/work") / "mh0492" / "m301067" / "hackaton25" / "curtains" / model_year / model_month / model_day
    curtain_label = f"{model_year}{model_month}{model_day}_{model}_zoom{zoom}"
    curtain_file = curtain_dir / f"ec_curtains_{curtain_label}.zarr"

    # Load catalog and dataset at time nearest model time given
    cat = intake.open_catalog(
        "https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml"
    )[current_location]
    model_datetime = np.datetime64(f"{model_year}-{model_month}-{model_day}")
    ds = cat[model](zoom=zoom).to_dask().sel(time=model_datetime, method="nearest")
    if nlevels_coarsen > 0:
        ds = coarsen_dataset(ds, nlevels_coarsen)

    # Create curtain dataset for model along points of track in ec_file
    ds_curtains = create_curtains_dataset(model, zoom, ds, ec_files, weights_dir)
    ds_curtains.attrs.update({
        "model_datetime": str(model_datetime),
        "ec_track_date": ec_year + ec_month + ec_day,
        "ec_track_date_format": 'YYYYMMDD'
    })

    # write out .zarr curtain dataset
    curtain_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing curtains dataset in {curtain_dir}")
    if curtain_file.is_dir():
        print(f"WARNING: overwriting exisiting zarr dataset: {curtain_file}")
        shutil.rmtree(curtain_file)
    ds_curtains.to_zarr(curtain_file)
    print(f"Curtain extracted and saved in {curtain_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="script to write .zarr curtain dataset for a day")
    parser.add_argument('model', help="name of the model, e.g. 'icon_d3hp003'")
    parser.add_argument('zoom', type=int, help="zoom level")
    parser.add_argument('date', help="earthcare track date in format 'YYYY/MM/DD'")
    parser.add_argument('--nlevels_coarsen', type=int,
                        help="number of zoom levels to coarsen dataset by",
                        default=0)
    args = parser.parse_args()
    write_curtain(args.model, args.zoom, args.date, nlevels_coarsen=args.nlevels_coarsen)
