"""
Fetches and caches a CMEMS L4 gridded SLA product for a given date range,
per a data_l4 config -- the L4 analog of build_experiment_data.py.

Unlike build_experiment_data.py, this is fetch-only: L4 is a pre-gridded
"map of the earth at specific times", not point data to be split/normalized/
tensorized for training, so there's no split/tensorize step here.

Usage:
    python build_l4_data.py --config configs/data_l4/DUACS/l4_DUACS_my_4_weeks.yaml [--plot]

Runs from the main venv (see copernicus_l4_pipeline.py's module docstring
for why this product doesn't need the separate fetch venv that
build_experiment_data.py requires).
"""

import argparse

import yaml
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from copernicus_l4_pipeline import fetch_and_store_l4, load_l4_dataset, zarr_path_for_config
from drive_paths import get_drive_base_path

# Whole-globe grid snapshot resolution + figure sizing -- kept byte-for-byte
# in sync with experiment_5.py's/replot_grid.py's own copies of these same
# constants/helper (each file keeps its own duplicate rather than sharing
# one, per existing project convention -- see replot_grid.py's module
# docstring). The L4 product happens to natively ship on exactly this same
# 0.125deg/2880x1440 grid (verified live 2026-08-19), so no
# resampling/regridding is needed to match exp5's own output resolution.
_NUM_LONGS = 2880  # 0.125 deg/pixel: 360 / 0.125
_NUM_LATS = 1440   # 0.125 deg/pixel: 180 / 0.125
_GRID_DPI = 256
_COLORBAR_HEIGHT_PX = 256  # legend space only, not pixel-critical
_BORDER_PX = 32  # small uniform white margin around the whole saved image

_TOTAL_WIDTH_PX = _NUM_LONGS + 2 * _BORDER_PX
_TOTAL_HEIGHT_PX = _NUM_LATS + _COLORBAR_HEIGHT_PX + 2 * _BORDER_PX
_FIG_WIDTH_IN = _TOTAL_WIDTH_PX / _GRID_DPI
_FIG_HEIGHT_IN = _TOTAL_HEIGHT_PX / _GRID_DPI

# Matches the fixed -0.25/0.25 SLA colour scale used for final_mean.png in
# experiment_5.py/replot_grid.py, so an L4 snapshot and a model prediction
# snapshot are visually comparable side by side.
_SLA_VMIN = -0.25
_SLA_VMAX = 0.25


def _save_global_grid_plot(data_np, cmap, vmin, vmax, colorbar_label, save_path, data_extent=(-180, 180, -90, 90)):
    """Saves a global snapshot with the map view fixed to the whole globe
    (matching experiment_5.py's final_mean.png framing) but the image itself
    placed at `data_extent` -- defaults to the whole globe too, but must be
    passed explicitly for a bbox-restricted fetch, since data_np then only
    covers a sub-region and stretching it across the full [-180,180,-90,90]
    extent (this function's original bug, caught 2026-08-19 testing a bbox
    config) silently mislocates/distorts it. Only a full-globe data_extent
    is pixel-exact (no resampling) at _NUM_LONGS x _NUM_LATS -- a smaller
    extent is still resampled by matplotlib to fit its pixel footprint, same
    as any other imshow call."""
    fig = plt.figure(figsize=(_FIG_WIDTH_IN, _FIG_HEIGHT_IN), dpi=_GRID_DPI)
    map_left = _BORDER_PX / _TOTAL_WIDTH_PX
    map_width = _NUM_LONGS / _TOTAL_WIDTH_PX
    map_bottom = (_BORDER_PX + _COLORBAR_HEIGHT_PX) / _TOTAL_HEIGHT_PX
    map_height = _NUM_LATS / _TOTAL_HEIGHT_PX
    ax = fig.add_axes(
        [map_left, map_bottom, map_width, map_height],
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    im = ax.imshow(
        data_np,
        origin="lower",
        cmap=cmap,
        extent=data_extent,
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    cax_left = map_left + 0.15 * map_width
    cax_width = 0.7 * map_width
    cax_bottom = _BORDER_PX / _TOTAL_HEIGHT_PX + 0.35 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
    cax_height = 0.3 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=colorbar_label)
    fig.savefig(save_path, dpi=_GRID_DPI)
    plt.close(fig)


def _save_overview_plot(ds, variable, l4_product, save_path, time_index=None):
    """Saves a single day's global `variable` snapshot, at the same
    2880x1440/bordered/no-resampling rendering as experiment_5.py's
    final_mean.png, so it can be compared directly against a model's own
    whole-globe prediction. Defaults to the day at the midpoint of the
    fetched range (rather than the first day) so a multi-week config's plot
    is representative of the whole range, not just its very first day."""
    if time_index is None:
        time_index = ds.sizes["time"] // 2
    day = ds.isel(time=time_index)
    data_np = day[variable].values  # (latitude, longitude), ascending both axes
    # [:16] keeps date + HH:MM (e.g. "2025-06-01T00:00") -- read from the
    # real timestamp rather than hardcoding "00:00", so this stays correct
    # even if a future L4 product's daily grids aren't stamped at exact
    # midnight the way DUACS's are.
    time_str = str(day["time"].values)[:16].replace("-", "/").replace("T", " ")

    # Pixel-center coordinates -> cell-edge extent (each pixel's true
    # footprint is +/- half a grid step past its center coordinate). Using
    # the data's own lon/lat bounds here -- rather than a hardcoded
    # [-180,180,-90,90] -- is what keeps a bbox-restricted fetch correctly
    # geolocated instead of stretched across the whole globe.
    lon, lat = day["longitude"].values, day["latitude"].values
    half_lon, half_lat = (lon[1] - lon[0]) / 2, (lat[1] - lat[0]) / 2
    data_extent = (lon[0] - half_lon, lon[-1] + half_lon, lat[0] - half_lat, lat[-1] + half_lat)

    _save_global_grid_plot(
        data_np, cmap="coolwarm", vmin=_SLA_VMIN, vmax=_SLA_VMAX,
        colorbar_label=f"{l4_product} {variable.upper()} (m) -- {time_str}", save_path=save_path,
        data_extent=data_extent,
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch and cache a CMEMS L4 gridded SLA product")
    parser.add_argument("--config", type=str, required=True, help="Path to a data_l4 config YAML file")
    parser.add_argument("--plot", action="store_true", help="Save a single-day global overview PNG alongside the Zarr store on Drive")
    parser.add_argument("--force-redownload", action="store_true", help="Re-fetch even if a cached Zarr store already exists")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Fetching L4 grid for '{config['name']}' ({config['start_date']} to {config['end_date']})...")
    fetch_and_store_l4(config, force_redownload=args.force_redownload)

    if args.plot:
        ds = load_l4_dataset(config)
        drive_base, _ = get_drive_base_path()
        zarr_path, _, _ = zarr_path_for_config(config, drive_base)
        variable = config["variables"][0]
        l4_product = config.get("l4_product", "DUACS")
        save_path = zarr_path.parent / f"{zarr_path.stem}_overview.png"
        _save_overview_plot(ds, variable, l4_product, save_path)
        print(f"Saved overview plot to {save_path}")


if __name__ == "__main__":
    main()
