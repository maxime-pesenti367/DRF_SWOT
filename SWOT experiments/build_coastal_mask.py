"""
Builds the coastal-distance component of the regional evaluation mask (Step 1
of regional_mask_spec.md): downloads NASA OBPG's global "distance to nearest
coastline" grid, regrids it onto the same target grid as
build_variance_mask.py, and thresholds it at COASTAL_DISTANCE_THRESHOLD_KM.

The NASA docs page linked from the original spec
(oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/) now 301-redirects to
a generic OB.DAAC landing page with no trace of the dataset -- checked
2026-08-27. The data itself is still live, mirrored on PacIOOS's ERDDAP
server, and is downloaded here as a plain HTTP griddap request -- no auth,
no copernicusmarine involved. The raw grid is cached locally so repeat runs
don't re-download it (~160MB).

Native resolution is 0.04deg (9000x4500) -- finer than the 0.125deg target
grid, so bilinear interpolation downsamples it (per spec: "this field is
smooth at these scales", no area-weighting needed). Target grid coordinates
are read directly from the DUACS L4 Zarr store via the same data_l4 config
used by build_variance_mask.py, rather than hardcoded, so this mask lands on
exactly the same grid -- pixel-for-pixel alignment between the two component
masks is what a future combine step (spec step 3, not yet implemented) will
need.

The raw dataset's own metadata declares _FillValue=0.0 for `dist`, but 0.0
is also the genuine value for a pixel sitting exactly on the coastline (the
ERDDAP metadata is simply wrong here, not describing an actual gap in the
grid -- it's documented as full global coverage with no missing cells). If
loaded with the default mask_and_scale=True, xarray would silently turn
every true zero (i.e. the coastline itself) into NaN. We open with
mask_and_scale=False to keep those values intact.

Usage:
    python build_coastal_mask.py --config configs/data_l4/DUACS/l4_DUACS_my_2025.yaml
"""

import argparse
import shutil
import ssl
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import xarray as xr
import yaml
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from copernicus_l4_pipeline import load_l4_dataset

# DUACS QUID convention threshold (Pujol et al. 2016) -- cells within this
# distance of the coast are classed coastal, overriding the variance split.
COASTAL_DISTANCE_THRESHOLD_KM = 200.0

_MASKS_DIR = Path(__file__).parent / "masks"

# Plot colours, indexed to coastal's -1/0/1 categories (land, offshore, coastal).
_LAND_COLOR = "#b0b0b0"
_COASTAL_COLOR = "#dd8452"
_OFFSHORE_COLOR = "#4c72b0"
_RAW_DIST2COAST_URL = (
    "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dist2coast_4deg.nc"
    "?dist[0:1:4499][0:1:8999]"
)
_RAW_DIST2COAST_PATH = _MASKS_DIR / "dist2coast_4deg_raw.nc"


def _download_dist2coast(force_redownload: bool = False) -> Path:
    if _RAW_DIST2COAST_PATH.exists() and not force_redownload:
        print(f"Using cached raw dist2coast grid at {_RAW_DIST2COAST_PATH}")
        return _RAW_DIST2COAST_PATH

    _MASKS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading global dist2coast grid (0.04deg, ~160MB) from {_RAW_DIST2COAST_URL} ...")
    # Explicit certifi CA bundle -- the stdlib ssl default context can't find
    # a usable local issuer cert on some Windows python.org installs
    # (SSLCertVerificationError), unrelated to this being the wrong URL/host.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(_RAW_DIST2COAST_URL, context=ssl_context) as response, \
            open(_RAW_DIST2COAST_PATH, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Saved raw grid to {_RAW_DIST2COAST_PATH}")
    return _RAW_DIST2COAST_PATH


def build_coastal_mask(config: dict, force_redownload: bool = False) -> xr.Dataset:
    raw_path = _download_dist2coast(force_redownload=force_redownload)
    raw = xr.open_dataset(raw_path, mask_and_scale=False)  # see module docstring re: _FillValue=0.0

    target = load_l4_dataset(config)
    target_lat = target["latitude"].values
    target_lon = target["longitude"].values

    print(f"Regridding dist2coast from {raw.sizes['longitude']}x{raw.sizes['latitude']} "
          f"(0.04deg) to {target_lon.size}x{target_lat.size} (target grid), bilinear...")
    distance_km = raw["dist"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    distance_km = distance_km.astype(np.float32)
    distance_km.name = "distance_km"
    distance_km.attrs = {
        "units": "km",
        "long_name": "distance to nearest coastline (negative = over land)",
        "source": "NASA OBPG dist2coast, 0.04deg, bilinearly regridded",
    }

    coastal = xr.where(
        distance_km < 0, -1,
        xr.where(distance_km < COASTAL_DISTANCE_THRESHOLD_KM, 1, 0),
    ).astype(np.int8)
    coastal.name = "coastal"
    coastal.attrs = {
        "long_name": "-1 = land, 0 = offshore (>= threshold_km), 1 = coastal (< threshold_km)",
        "threshold_km": COASTAL_DISTANCE_THRESHOLD_KM,
    }

    return xr.merge([distance_km, coastal]).load()


def _save_coastal_mask_plot(coastal: xr.DataArray, save_path: Path):
    cmap = ListedColormap([_LAND_COLOR, _OFFSHORE_COLOR, _COASTAL_COLOR])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    lon, lat = coastal["longitude"].values, coastal["latitude"].values
    half_lon, half_lat = (lon[1] - lon[0]) / 2, (lat[1] - lat[0]) / 2
    extent = (lon[0] - half_lon, lon[-1] + half_lon, lat[0] - half_lat, lat[-1] + half_lat)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88], projection=ccrs.PlateCarree())
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.imshow(coastal.values, origin="lower", cmap=cmap, norm=norm, extent=extent,
              transform=ccrs.PlateCarree(), interpolation="none")
    ax.coastlines(linewidth=0.5)
    ax.legend(handles=[
        mpatches.Patch(color=_LAND_COLOR, label="Land"),
        mpatches.Patch(color=_COASTAL_COLOR, label=f"Coastal (< {COASTAL_DISTANCE_THRESHOLD_KM:.0f} km)"),
        mpatches.Patch(color=_OFFSHORE_COLOR, label=f"Offshore (≥ {COASTAL_DISTANCE_THRESHOLD_KM:.0f} km)"),
    ], loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_title("Coastal distance mask")
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build the coastal-distance component of the regional evaluation mask")
    parser.add_argument("--config", type=str, required=True, help="Path to a data_l4 config YAML providing the target grid (its Zarr store must already be fetched)")
    parser.add_argument("--output", type=str, default=None, help="Output .nc path (default: masks/coastal_mask.nc)")
    parser.add_argument("--force-redownload", action="store_true", help="Re-download the raw dist2coast grid even if a cached copy exists")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_ds = build_coastal_mask(config, force_redownload=args.force_redownload)

    output_path = Path(args.output) if args.output else _MASKS_DIR / "coastal_mask.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(output_path)

    coastal = out_ds["coastal"]
    n_total = int(coastal.size)
    n_land = int((coastal == -1).sum())
    n_offshore = int((coastal == 0).sum())
    n_coastal = int((coastal == 1).sum())
    print(f"Saved coastal mask to {output_path}")
    print(f"Land: {n_land}/{n_total} ({100 * n_land / n_total:.1f}%), "
          f"offshore: {n_offshore}/{n_total} ({100 * n_offshore / n_total:.1f}%), "
          f"coastal: {n_coastal}/{n_total} ({100 * n_coastal / n_total:.1f}%)")

    plot_path = output_path.with_suffix(".png")
    _save_coastal_mask_plot(coastal, plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
