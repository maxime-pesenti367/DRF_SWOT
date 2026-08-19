"""
Fetch/store/load pipeline for gridded L4 sea level products -- the
pre-gridded counterpart to the per-satellite L3 along-track products handled
by copernicus_pipeline.py.

Unlike L3, this is a single regular (time, latitude, longitude) grid, not a
sparse per-satellite trajectory table -- no per-satellite loop, no
long-format pivot needed.

Only the CMEMS/DUACS L4 product (SEALEVEL_GLO_PHY_L4_MY_008_047 /
SEALEVEL_GLO_PHY_L4_NRT_008_046) is implemented so far -- configs select it
via `l4_product: DUACS`. Other L4 products (e.g. NASA MEaSUREs) are expected
to be added later as additional entries in L4_DATASET_MAP plus, if their
fetch mechanics differ from copernicusmarine, a branch in
fetch_and_store_l4/load_l4_dataset; _require_duacs() below is the single
place that currently rejects anything else, so it's the one spot to relax
once a second provider is actually implemented.

copernicusmarine.open_dataset() (a lazy remote accessor) already returns
data in the (time, latitude, longitude) shape this module wants, via
Copernicus's ARCO/Zarr backend, so there's no NetCDF/subset()-writer bug to
work around here (that bug -- see copernicus_pipeline.py's
_sparse_dataframe_to_dataset docstring -- is specific to sparse/SQLite-backed
altimetry products). Verified this L4 product opens fine even from the
older, numpy==1.26.4-compatible copernicusmarine already pinned in the main
requirements.txt (copernicusmarine.describe(contains=["SEALEVEL_GLO_PHY_L4"])
+ a live open_dataset() call against both the MY and NRT dataset IDs below,
checked 2026-08-19). That means, unlike build_experiment_data.py, this
module runs fine from the main venv -- no separate fetch venv needed.
"""

import shutil

import numpy as np
import xarray as xr
import copernicusmarine

from drive_paths import get_drive_base_path


# l4_product -> product_type ("my"/"nrt") -> Copernicus dataset ID for the
# daily, 0.125deg all-satellite DUACS L4 SLA grid. Confirmed live via
# copernicusmarine.describe(contains=["SEALEVEL_GLO_PHY_L4"]) on 2026-08-19 --
# each product also has a monthly-mean and/or demo/interim variant not used
# here. Nested one level by l4_product so a later provider (e.g. "MEASURES")
# can be added as a sibling key without touching the "DUACS" entries.
L4_DATASET_MAP = {
    "DUACS": {
        "my": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        "nrt": "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
    },
}


def _require_duacs(l4_product):
    """Only DUACS is wired up so far -- see module docstring. Raises for
    anything else instead of silently mishandling it (e.g. a future
    "MEASURES" config would need a different fetch mechanism entirely, not
    just a different dataset ID)."""
    if l4_product != "DUACS":
        raise NotImplementedError(
            f"L4 product '{l4_product}' is not yet supported -- only 'DUACS' "
            f"is currently implemented."
        )


def zarr_path_for_config(config, drive_base):
    """Returns (zarr_path, l4_product, product_type). Every date range gets
    its own <start_date>_<end_date> folder (rather than all ranges' .zarr/
    .png pairs landing flat in the same product_type folder), so
    build_l4_data.py's overview PNG -- saved as a sibling of the .zarr store
    via zarr_path.parent -- ends up grouped with it instead of the folder
    listing growing one .zarr + one .png per fetch."""
    l4_product = config.get("l4_product", "DUACS")
    product_type = config.get("product_type", "my")
    start_date = config["start_date"].replace(":", "-")
    end_date = config["end_date"].replace(":", "-")
    timespan_dir = f"{start_date}_{end_date}"
    zarr_name = f"l4_sla_{start_date}_{end_date}.zarr"
    zarr_path = drive_base / "zarr files" / "l4" / l4_product / product_type / timespan_dir / zarr_name
    return zarr_path, l4_product, product_type


# Downloads and stores the L4 grid for a data config's date range/variables
# as a Zarr store on Google Drive.
def fetch_and_store_l4(config: dict, force_redownload: bool = False):

    drive_base, temp_dir = get_drive_base_path()
    final_drive_path, l4_product, product_type = zarr_path_for_config(config, drive_base)
    _require_duacs(l4_product)

    if final_drive_path.exists() and not force_redownload:
        print(f"Skipping L4 fetch: dataset already exists at {final_drive_path}")
        return

    if product_type not in L4_DATASET_MAP[l4_product]:
        raise ValueError(f"Unknown L4 product_type '{product_type}' (expected 'my' or 'nrt')")
    dataset_id = L4_DATASET_MAP[l4_product][product_type]

    bbox = config.get("bbox", {})

    print(f"\n--- Fetching L4 grid ({l4_product}/{product_type}) from Copernicus ---")
    print("Opening remote dataset...")
    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=config["variables"],
        start_datetime=config["start_date"],
        end_datetime=config["end_date"],
        minimum_longitude=bbox.get("lon_min"),
        maximum_longitude=bbox.get("lon_max"),
        minimum_latitude=bbox.get("lat_min"),
        maximum_latitude=bbox.get("lat_max"),
    )

    # The remote dataset's variables carry zarr-v2-style encoding
    # (numcodecs Blosc compressors) inherited from Copernicus's own backing
    # store -- writing that straight through xarray's zarr-v3 writer fails
    # with "Expected a BytesBytesCodec, got <Blosc>" (verified live
    # 2026-08-19). Clearing it lets xarray/zarr pick fresh v3-native
    # defaults for our own local store; we don't need to preserve
    # Copernicus's original chunking/compression choices.
    for var in list(ds.variables):
        ds[var].encoding = {}

    # Stage the Zarr write in a local temp folder before copying to the
    # (often network-mounted) Drive path -- avoids a partially-written store
    # landing directly on Drive if this fails partway through. Same pattern
    # as fetch_and_store_satellites in copernicus_pipeline.py.
    temp_zarr_dir = temp_dir / "zarr_l4"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading and writing to local Zarr store...")
    ds.to_zarr(temp_zarr_dir, mode="w")
    ds.close()

    final_drive_path.parent.mkdir(parents=True, exist_ok=True)
    if final_drive_path.exists():
        shutil.rmtree(final_drive_path)

    print("Transferring to Google Drive...")
    shutil.copytree(temp_zarr_dir, final_drive_path)

    shutil.rmtree(temp_dir)
    print(f"Saved L4 grid to {final_drive_path}")


# Loads the cached L4 Zarr store lazily, mirroring
# copernicus_pipeline.load_experiment_dataset.
def load_l4_dataset(config: dict) -> xr.Dataset:

    drive_base, _ = get_drive_base_path()
    zarr_path, l4_product, _ = zarr_path_for_config(config, drive_base)
    _require_duacs(l4_product)

    if not zarr_path.exists():
        raise FileNotFoundError(
            f"No saved L4 data found at {zarr_path}. Run build_l4_data.py "
            f"with this config first."
        )

    return xr.open_zarr(zarr_path)


# Nearest-neighbour lookup of `variable` at a single (longitude, latitude,
# time) point. xr.open_zarr is lazy -- .sel(..., method="nearest") only
# reads the one chunk containing the matched grid cell off disk/Drive, not
# the whole store, so this is cheap to call repeatedly.
def query_l4_point(ds: xr.Dataset, longitude: float, latitude: float, time, variable: str = "sla") -> float:
    point = ds[variable].sel(longitude=longitude, latitude=latitude, time=time, method="nearest")
    return float(point.load())


# Vectorized nearest-neighbour lookup for many (lon, lat, time) points at
# once -- e.g. pulling L4 ground-truth values at a DRF test set's query
# locations for comparison. longitudes/latitudes/times must be equal-length
# 1D array-likes; times must be parseable by pandas.to_datetime. Using
# xr.DataArray indexers sharing a "points" dim makes this a pointwise
# lookup (one value per input point) rather than xarray's default outer-
# product behaviour (every lon x every lat x every time).
def query_l4_points(ds: xr.Dataset, longitudes, latitudes, times, variable: str = "sla") -> np.ndarray:
    lon_da = xr.DataArray(longitudes, dims="points")
    lat_da = xr.DataArray(latitudes, dims="points")
    time_da = xr.DataArray(times, dims="points")
    points = ds[variable].sel(longitude=lon_da, latitude=lat_da, time=time_da, method="nearest")
    return points.load().values
