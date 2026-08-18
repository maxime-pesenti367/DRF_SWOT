import os
import sys
import shutil
import platform
from pathlib import Path
import xarray as xr
import copernicusmarine

import torch
from torch.utils.data import TensorDataset, random_split

import pandas as pd
import numpy as np

from drive_paths import get_drive_base_path  # noqa: F401 -- re-exported for existing callers (build_experiment_data.py, notebooks)


# Maps product type (nrt = near-real-time, my = multi-year reprocessed) ->
# clean satellite name -> resolution -> Copernicus product ID. Satellite
# names are kept consistent across "nrt" and "my" where they refer to the
# same physical satellite/orbit (e.g. "cryosat2" is the current/new-orbit
# product in both), so switching product_type in a data config without
# changing the satellite list behaves predictably.
SATELLITE_DATASET_MAP = {
    "nrt": {
        "swot_nadir": {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_swon-l3-duacs_PT1S",
            "5hz": "cmems_obs-sl_glo_phy-ssh_nrt_swon-l3-duacs_PT0.2S"
        },
        "jason3_interleaved": {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_j3n-l3-duacs_PT1S",
            "5hz": "cmems_obs-sl_glo_phy-ssh_nrt_j3n-l3-duacs_PT0.2S"
        },
        "jason3_LRO": {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_j3g-l3-duacs_PT1S-i",
            "5hz": "cmems_obs-sl_glo_phy-ssh_nrt_j3g-l3-duacs_PT0.2S-i"
        },
        "cryosat2":  {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_c2n-l3-duacs_PT1S"
        },
        "sentinel3a":  {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_s3a-l3-duacs_PT1S",
            "5hz": "cmems_obs-sl_glo_phy-ssh_nrt_s3a-l3-duacs_PT0.2S"
        },
        "sentinel3b":  {
            "1hz": "cmems_obs-sl_glo_phy-ssh_nrt_s3b-l3-duacs_PT1S",
            "5hz": "cmems_obs-sl_glo_phy-ssh_nrt_s3b-l3-duacs_PT0.2S"
        }
        # PLEASE ADD MORE AS NEEDED
    },
    "my": {
        # Only 1hz (PT1S) products exist for the MY line as of the codes
        # supplied for this project -- no 5hz/PT0.2S variants.
        "topex_poseidon":               {"1hz": "cmems_obs-sl_glo_phy-ssh_my_tp-l3-duacs_PT1S"},
        "topex_poseidon_new_orbit":     {"1hz": "cmems_obs-sl_glo_phy-ssh_my_tpn-l3-duacs_PT1S"},
        "ers1":                         {"1hz": "cmems_obs-sl_glo_phy-ssh_my_e1-l3-duacs_PT1S"},
        "ers1_geodetic":                {"1hz": "cmems_obs-sl_glo_phy-ssh_my_e1g-l3-duacs_PT1S"},
        "ers2":                         {"1hz": "cmems_obs-sl_glo_phy-ssh_my_e2-l3-duacs_PT1S"},
        "gfo":                          {"1hz": "cmems_obs-sl_glo_phy-ssh_my_g2-l3-duacs_PT1S"},
        "envisat":                      {"1hz": "cmems_obs-sl_glo_phy-ssh_my_en-l3-duacs_PT1S"},
        "envisat_new_orbit":            {"1hz": "cmems_obs-sl_glo_phy-ssh_my_enn-l3-duacs_PT1S"},
        "jason1":                       {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j1-l3-duacs_PT1S"},
        "jason1_new_orbit":             {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j1n-l3-duacs_PT1S"},
        "jason1_geodetic":              {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j1g-l3-duacs_PT1S"},
        "jason2":                       {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j2-l3-duacs_PT1S"},
        "jason2_interleaved":           {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j2n-l3-duacs_PT1S"},
        "jason2_long_repeat":           {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j2g-l3-duacs_PT1S"},
        "jason3":                       {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j3-l3-duacs_PT1S"},
        "jason3_interleaved":           {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j3n-l3-duacs_PT1S"},
        "jason3_LRO":                   {"1hz": "cmems_obs-sl_glo_phy-ssh_my_j3g-l3-duacs_PT1S-i"},
        "cryosat2_old_orbit":           {"1hz": "cmems_obs-sl_glo_phy-ssh_my_c2-l3-duacs_PT1S"},
        "cryosat2":                     {"1hz": "cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S"},
        "saral_altika":                 {"1hz": "cmems_obs-sl_glo_phy-ssh_my_al-l3-duacs_PT1S"},
        "saral_altika_geodetic":        {"1hz": "cmems_obs-sl_glo_phy-ssh_my_alg-l3-duacs_PT1S"},
        "haiyang2a":                    {"1hz": "cmems_obs-sl_glo_phy-ssh_my_h2a-l3-duacs_PT1S"},
        "haiyang2a_geodetic":           {"1hz": "cmems_obs-sl_glo_phy-ssh_my_h2ag-l3-duacs_PT1S"},
        "haiyang2b":                    {"1hz": "cmems_obs-sl_glo_phy-ssh_my_h2b-l3-duacs_PT1S"},
        "sentinel3a":                   {"1hz": "cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S"},
        "sentinel3b":                   {"1hz": "cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S"},
        "sentinel6a_lrm":               {"1hz": "cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S"},
        "swot_nadir":                   {"1hz": "cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S"},
        "swot_nadir_calval":            {"1hz": "cmems_obs-sl_glo_phy-ssh_my_swonc-l3-duacs_PT1S"},
        # PLEASE ADD MORE AS NEEDED
    },
}


def _resolve_satellites(config, product_type):
    """Returns the list of {"name", "freq"} dicts to fetch/load for this
    config. `satellites: all` expands to every satellite registered under
    this product_type, all at the single `freq` the config specifies at the
    top level (config["freq"]) -- an explicit list is passed through
    unchanged (each entry keeps its own per-satellite freq, as before)."""
    satellites = config["satellites"]
    if satellites == "all":
        freq = config["freq"]
        return [{"name": name, "freq": freq} for name in SATELLITE_DATASET_MAP[product_type]]
    return satellites


# Downloads and updates per-satellite Zarr stores on Google Drive
# based on a provided DatasetSpec/Config
def fetch_and_store_satellites(config: dict, force_redownload: bool = False):

    drive_base, temp_dir = get_drive_base_path()
    drive_zarr_base = drive_base / "zarr files"
    drive_zarr_base.mkdir(parents=True, exist_ok=True)

    product_type = config.get("product_type", "nrt")
    satellites = _resolve_satellites(config, product_type)
    start_date = config["start_date"]
    end_date = config["end_date"]
    start_date = start_date.replace(":", "-")
    end_date = end_date.replace(":", "-")
    variables = config["variables"]

    for sat in satellites:
        if isinstance(sat, dict):
            sat_name = sat["name"]
            freq = sat.get("freq", "1hz")
        else:
            sat_name = sat
            freq = "1hz"

        if sat_name not in SATELLITE_DATASET_MAP[product_type]:
            print(f"Skipping unknown satellite: {sat_name} (product_type={product_type})")
            continue
        if freq not in SATELLITE_DATASET_MAP[product_type][sat_name]:
            print(f"Skipping {sat_name}: no {freq} product for product_type={product_type}")
            continue

        dataset_id = SATELLITE_DATASET_MAP[product_type][sat_name][freq]

        zarr_name = f"{sat_name}_{freq}_{start_date}_{end_date}.zarr"

        final_drive_path = drive_zarr_base / product_type / sat_name / zarr_name

        if final_drive_path.exists() and not force_redownload:
            print(f"Skipping {sat_name}: Dataset already exists at {final_drive_path}")
            continue

        print(f"\n--- Fetching {sat_name} ({freq}) from Copernicus ---")
        
        # 1. Prepare temporary folders for NetCDF and Zarr
        temp_nc_dir = temp_dir / "raw_nc"
        temp_zarr_dir = temp_dir / "zarr"
        
        # Clear them if they already exist from a previous failed run
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_nc_dir.mkdir(parents=True, exist_ok=True)

        # 2. Download the NetCDF file(s) locally using .subset()
        print("Downloading NetCDF file(s)...")
        
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                start_datetime=start_date,
                end_datetime=end_date,
                output_directory=str(temp_nc_dir),
                file_format="netcdf"
            )
        except Exception as e:
            print(f"Warning: Could not fetch {sat_name}. It likely has no data for this date range.")
            print(f"API Error: {e}")
            
            # Clean up the temp directory so it's empty for the next satellite in the loop!
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            # Skip the rest of the loop and move to the next satellite
            continue

        # 3. Load the downloaded NetCDF and convert to Zarr
        # rglob() recursively searches through all subfolders!
        # It will find the .nc files no matter what folders the API created.
        nc_files = list(temp_nc_dir.rglob("*.nc"))
        
        if not nc_files:
            print(f"Error: No NetCDF files found anywhere inside {temp_nc_dir}.")
            continue
            
        print(f"Found {len(nc_files)} NetCDF file(s). Converting to Zarr...")
        
        # open_mfdataset safely handles 1 file or multiple partitioned files
        ds = xr.open_mfdataset(nc_files, combine='by_coords')
        ds.to_zarr(temp_zarr_dir, mode='w')
        
        # Always close the dataset to free up memory and release file locks
        ds.close()

        # 4. Transfer the final Zarr folder to Google Drive
        final_drive_path.parent.mkdir(parents=True, exist_ok=True)
        if final_drive_path.exists():
            shutil.rmtree(final_drive_path)

        print("Transferring to Google Drive...")
        shutil.copytree(temp_zarr_dir, final_drive_path)
        
        # Clean up the temporary SSD files
        shutil.rmtree(temp_dir)
        print(f"Saved {sat_name} to {final_drive_path}")


# Loads and returns a dictionary of xarray Datasets for each satellite 
# in the experiment config, directly from Google Drive.
def load_experiment_dataset(config: dict):

    drive_base, _ = get_drive_base_path()
    drive_zarr_base = drive_base / "zarr files"
    product_type = config.get("product_type", "nrt")
    start_date = config["start_date"]
    end_date = config["end_date"]
    start_date = start_date.replace(":", "-")
    end_date = end_date.replace(":", "-")

    loaded_data = {}
    for sat_dict in _resolve_satellites(config, product_type):
        sat_name = sat_dict["name"]
        freq = sat_dict.get("freq", "1hz") # Defaults to 1hz if not specified

        zarr_path = drive_zarr_base / product_type / sat_name / f"{sat_name}_{freq}_{start_date}_{end_date}.zarr"
        if not zarr_path.exists():
            # Instead of raising an error, print warning and skip
            print(f"Warning: No saved data found for {sat_name} in this date range. Skipping.")
            continue

        # Open Zarr lazily
        loaded_data[sat_name] = xr.open_zarr(zarr_path)
        
    return loaded_data


# Returns clean Pandas df ready for PyTorch OR returns combined Xarray for plotting
# Takes a dictionary of satellite Xarray datasets, assigns satellite labels, 
# and combines them into a single structure.
# output_format: 'xarray' (best for plotting) or 'pandas' (best for ML conversion)
def combine_for_drf(data_dict, output_format):

    if output_format not in ["xarray", "pandas"]:
        raise Exception(f"output_format is not xarray or pandas")

    datasets_to_combine = []
    
    for sat_name, ds in data_dict.items():
        # Add the satellite label
        labeled_ds = ds.assign_coords(satellite=("time", [sat_name] * ds.sizes["time"]))
        datasets_to_combine.append(labeled_ds)
        
    # Concatenate all into one Xarray dataset
    combined_ds = xr.concat(datasets_to_combine, dim="time").sortby("time")
    
    if output_format == "pandas":
        # Drop NaNs immediately to save memory during ML training
        return combined_ds.to_dataframe().reset_index().dropna(subset=['sla_filtered', 'latitude', 'longitude'])
    
    return combined_ds



# Takes pandas df and processes and splits data, returns Pytorch tensor (.pt)
# Usage:
# processed_tensors = process_and_split_dataframe(your_combined_df)
# torch.save(processed_tensors, 'ready_for_drf.pt')

def process_and_split_dataframe(df, spatial_cols=['longitude', 'latitude'], temporal_col=['time'], target_col='sla_filtered', train_ratio=0.8):
    df_processed = df.copy()

    if pd.api.types.is_datetime64_any_dtype(df_processed[temporal_col[0]]):
        df_processed[temporal_col[0]] = df_processed[temporal_col[0]].astype('int64') / 10**9

    df_processed[spatial_cols] = np.deg2rad(df_processed[spatial_cols])

    spatial_X = df_processed[spatial_cols].to_numpy()
    temporal_X = df_processed[temporal_col].to_numpy()
    Y = df_processed[target_col].to_numpy()

    spatial_tensor = torch.tensor(spatial_X, dtype=torch.float32)
    temporal_tensor = torch.tensor(temporal_X, dtype=torch.float32)
    target_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset_size = len(spatial_tensor)
    train_size = int(train_ratio * dataset_size)
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    spatial_train_raw = spatial_tensor[train_indices]
    temporal_train_raw = temporal_tensor[train_indices]
    y_train = target_tensor[train_indices]
    
    spatial_test_raw = spatial_tensor[test_indices]
    temporal_test_raw = temporal_tensor[test_indices]
    y_test = target_tensor[test_indices]

    spatial_mean = spatial_train_raw.mean(dim=0)
    spatial_std = spatial_train_raw.std(dim=0)
    
    temporal_mean = temporal_train_raw.mean(dim=0)
    temporal_std = temporal_train_raw.std(dim=0)

    spatial_X_train = (spatial_train_raw - spatial_mean) / (spatial_std + 1e-8)
    temporal_X_train = (temporal_train_raw - temporal_mean) / (temporal_std + 1e-8)
    
    spatial_X_test = (spatial_test_raw - spatial_mean) / (spatial_std + 1e-8)
    temporal_X_test = (temporal_test_raw - temporal_mean) / (temporal_std + 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return {
        'spatial_X_train': spatial_X_train.to(device),
        'temporal_X_train': temporal_X_train.to(device),
        'y_train': y_train.to(device),
        'spatial_X_test': spatial_X_test.to(device),
        'temporal_X_test': temporal_X_test.to(device),
        'y_test': y_test.to(device),
        'normalization_stats': {
            'spatial_mean': spatial_mean, 'spatial_std': spatial_std,
            'temporal_mean': temporal_mean, 'temporal_std': temporal_std
        }
    }


# Save pytorch tensor in drive
def store_tensors(tensors, experiment_name):

    drive_base, _ = get_drive_base_path()
    drive_pt_base = drive_base / "pytorch tensors"

    final_pt_path = drive_pt_base / f"{experiment_name}.pt" 

    torch.save(tensors, final_pt_path)

    print(f"Saved {experiment_name} to {final_pt_path}")


