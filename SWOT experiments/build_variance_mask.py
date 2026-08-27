"""
Builds the offshore SLA-variance component of the regional evaluation mask
(Step 2 of regional_mask_spec.md) from an already-fetched CMEMS DUACS L4
grid. Reuses copernicus_l4_pipeline.load_l4_dataset() to read the cached
Zarr store built by build_l4_data.py -- this script does no fetching itself,
so the data_l4 config passed in must already have been fetched.

Per-grid-cell temporal variance of SLA is computed over the full reference
period with all months pooled (no deseasonalizing/detrending), matching the
DUACS QUID convention (Pujol et al. 2016; DUACS QUID CMEMS-SL-QUID-008-057).
The DUACS L4 product already ships on a 2880x1440 (0.125deg) grid -- the
same target grid used everywhere else in this project (see build_l4_data.py's
own comment, verified 2026-08-19) -- so no regridding happens here. See
build_coastal_mask.py for the counterpart mask, which does need to regrid
(its source is natively 0.04deg) and reads its target grid from the same
data_l4 config so both masks land on identical, pixel-aligned coordinates.

Runs fine from the main venv: importing copernicus_l4_pipeline pulls in
copernicusmarine, but only for its module-level import, not any network
call here -- see that module's docstring for why the numpy==1.26.4-pinned
copernicusmarine already in the main venv is sufficient for L4.

Usage:
    python build_variance_mask.py --config configs/data_l4/DUACS/l4_DUACS_my_2025.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from copernicus_l4_pipeline import load_l4_dataset

# DUACS QUID convention threshold (Pujol et al. 2016) -- offshore cells at or
# above this pooled-all-months SLA variance are classed high-variability.
SLA_VARIANCE_THRESHOLD_CM2 = 200.0

_MASKS_DIR = Path(__file__).parent / "masks"

# Plot colours, indexed to high_var's -1/0/1 categories (land, low_var, high_var).
_LAND_COLOR = "#b0b0b0"
_LOW_VAR_COLOR = "#4c72b0"
_HIGH_VAR_COLOR = "#c44e52"


def build_variance_mask(config: dict) -> xr.Dataset:
    ds = load_l4_dataset(config)
    sla_m = ds["sla"]

    print(f"Computing SLA variance over {sla_m.sizes['time']} days "
          f"({config['start_date']} to {config['end_date']}), all months pooled "
          f"(no deseasonalization) -- this reads the full time series, may take a while...")
    sla_variance_cm2 = (sla_m.var(dim="time") * (100.0 ** 2)).astype(np.float32)
    sla_variance_cm2.name = "sla_variance"
    sla_variance_cm2.attrs = {
        "units": "cm2",
        "long_name": "SLA temporal variance, all months pooled, no deseasonalization",
        "reference_period_start": config["start_date"],
        "reference_period_end": config["end_date"],
        "source_l4_config": config["name"],
    }

    high_var = xr.where(
        sla_variance_cm2.isnull(), -1,
        xr.where(sla_variance_cm2 >= SLA_VARIANCE_THRESHOLD_CM2, 1, 0),
    ).astype(np.int8)
    high_var.name = "high_var"
    high_var.attrs = {
        "long_name": "-1 = land/no data, 0 = offshore low-variability, 1 = offshore high-variability",
        "threshold_cm2": SLA_VARIANCE_THRESHOLD_CM2,
    }

    return xr.merge([sla_variance_cm2, high_var]).load()


def _save_variance_mask_plot(high_var: xr.DataArray, save_path: Path):
    cmap = ListedColormap([_LAND_COLOR, _LOW_VAR_COLOR, _HIGH_VAR_COLOR])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    lon, lat = high_var["longitude"].values, high_var["latitude"].values
    half_lon, half_lat = (lon[1] - lon[0]) / 2, (lat[1] - lat[0]) / 2
    extent = (lon[0] - half_lon, lon[-1] + half_lon, lat[0] - half_lat, lat[-1] + half_lat)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88], projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.imshow(high_var.values, origin="lower", cmap=cmap, norm=norm, extent=extent,
              transform=ccrs.PlateCarree(), interpolation="none")
    ax.coastlines(linewidth=0.5)
    ax.legend(handles=[
        mpatches.Patch(color=_LAND_COLOR, label="Land / no data"),
        mpatches.Patch(color=_LOW_VAR_COLOR, label=f"Offshore, low-var (< {SLA_VARIANCE_THRESHOLD_CM2:.0f} cm²)"),
        mpatches.Patch(color=_HIGH_VAR_COLOR, label=f"Offshore, high-var (≥ {SLA_VARIANCE_THRESHOLD_CM2:.0f} cm²)"),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("SLA variance mask")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build the SLA-variance component of the regional evaluation mask")
    parser.add_argument("--config", type=str, required=True, help="Path to the data_l4 config YAML whose fetched Zarr store to use")
    parser.add_argument("--output", type=str, default=None, help="Output .nc path (default: masks/sla_variance_mask.nc)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_ds = build_variance_mask(config)

    output_path = Path(args.output) if args.output else _MASKS_DIR / "sla_variance_mask.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(output_path)

    high_var = out_ds["high_var"]
    n_total = int(high_var.size)
    n_land = int((high_var == -1).sum())
    n_low = int((high_var == 0).sum())
    n_high = int((high_var == 1).sum())
    print(f"Saved SLA variance mask to {output_path}")
    print(f"Land/no data: {n_land}/{n_total} ({100 * n_land / n_total:.1f}%), "
          f"low-var: {n_low}/{n_total} ({100 * n_low / n_total:.1f}%), "
          f"high-var: {n_high}/{n_total} ({100 * n_high / n_total:.1f}%)")

    plot_path = output_path.with_suffix(".png")
    _save_variance_mask_plot(high_var, plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
