"""
Downloads raw SWOT L2 KaRIn Expert granules for a date range (via
podaac-data-downloader, see "how to download swot data.md" for the
underlying CLI and its netrc-based auth) and canonicalizes them into one
quality-filtered, corrected Parquet table -- the "clean SWOT truth" a
validation run compares DRF/DUACS against, built once per data_swot_l2
config rather than re-fetched/re-parsed every time.

Requires `pip install podaac-data-subscriber` (provides the
podaac-data-downloader CLI) in whichever venv runs this -- not part of the
numpy-conflicted fetch-venv split described in CLAUDE.md, since this has no
numpy dependency of its own; it just needs to be installed once.

Only the Expert product is supported: height_cor_xover (needed for the
corrected SSHA) only exists there, not in Basic/WindWave/Unsmoothed -- see
the Expert-vs-Basic discussion this project settled on earlier. Downloads
go through the dedicated SWOT_L2_LR_SSH_EXPERT_D collection (a
sub-collection of the combined SWOT_L2_LR_SSH_D used elsewhere in this
project, e.g. swot_expert_data_test.ipynb) rather than the combined
collection -- CMR filters to Expert-only server-side this way, with no
filename filter needed at all. Deliberately NOT the combined collection +
a -gr/-e filename filter: verified live that both silently return zero
granules when combined with a date range (-sd/-ed), even for a date
confirmed via the same query without the filter to have real matching
Expert files -- and the combined collection also serves Unsmoothed, whose
files are far larger, so a broken filter there risks downloading
everything instead of nothing. COLLECTION is hardcoded, not a config
field, since it should never vary per config.

Corrected SSHA = ssha_karin_2 + height_cor_xover, the same convention
already prototyped in swot_expert_data_test.ipynb. A point is kept only if:
  - ssha_karin_2 and height_cor_xover are both non-NaN (real per-pixel
    gaps in the swath, e.g. over land), and
  - ssha_karin_2_qual == max_qual_flag (default 0). ssha_karin_2_qual is a
    CF status_flag bitmask (see its flag_masks/flag_meanings attrs in the
    raw file); 0 means no issues flagged at all -- the strictest, safest
    default. Configurable via max_qual_flag if you want to loosen it later
    (e.g. to also keep "suspect_*" bits, just not "bad_*"/"degraded_*").
  - ancillary_surface_classification_flag == open_ocean_flag_value
    (default 0, i.e. open ocean only). This is NOT redundant with the
    quality flag above -- verified live on a real granule that
    ssha_karin_2_qual==0 alone still lets land/continental-water pixels
    through (one continental_water pixel reached ssha_karin_2 = +678m,
    vs. -2.8..+2.9m for the 302081 genuinely open-ocean qual==0 points in
    the same file), since ssha_karin_2_qual reflects instrument/processing
    quality, not surface type. Comparing DRF/DUACS (ocean-only products)
    against contaminated land/ice points would be meaningless anyway.

Each granule's data variables are (num_lines, num_pixels) -- an
along-track x cross-track swath, not a flat point list. time is 1D (one
per along-track line, shared by every cross-track pixel in that line) and
gets broadcast to match before flattening; latitude/longitude are already
2D. Longitude ships in [0, 360) in the raw file -- converted to [-180, 180]
here to match every other lon/lat convention already used in this project
(DUACS, copernicus_pipeline.py's L3 tracks).

lon/lat/swot_ssha are rounded to the raw file's own actual precision
(scale_factor: 1e-06 for latitude/longitude, 1e-04 for ssha_karin_2/
height_cor_xover) before saving -- discards zero real information, just
removes binary-floating-point representation noise from decoding scaled
integers (e.g. -115.85557700000001 -> -115.855577).

cycle_number/pass_number come from each granule's own global attributes
(not parsed from the filename), carried through as columns for provenance.

Raw granules land in data/karin/<name>/ -- auto-derived from the config's
name, never hand-typed, avoiding the copy-paste-bug class this project has
already hit twice elsewhere (see experiment_5.py's results_name comment).
podaac-data-downloader is invoked every run regardless of what's already
there -- it checks each granule's existence/checksum itself and only
(re-)fetches what's missing or doesn't match, --force overrides this to
re-download everything.

Usage:
    python build_swot_l2_data.py --config configs/data_swot_l2/swot_expert_jan2024.yaml
    python build_swot_l2_data.py --config configs/data_swot_l2/swot_expert_jan2024.yaml --force
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from drive_paths import get_drive_base_path

SCRIPT_DIR = Path(__file__).resolve().parent
# The Expert-only sub-collection (verified live via CMR: NOT the combined
# SWOT_L2_LR_SSH_D collection -- that one also serves Basic/WindWave/
# Unsmoothed, and combining it with either -gr or -e filtering to isolate
# Expert turned out to silently return zero results when also given a
# date range, even for dates confirmed to have real matching data. This
# collection filters server-side instead, so no filename filter is needed
# at all, and it never risks pulling down Unsmoothed's much larger files.
COLLECTION = "SWOT_L2_LR_SSH_EXPERT_D"


def _find_downloader_executable():
    """Locates the podaac-data-downloader console script installed by
    `pip install podaac-data-subscriber`. Checked relative to sys.executable
    first (where pip actually installs console scripts for this venv) since
    that directory isn't necessarily on PATH unless the venv is activated
    (this project's other scripts are typically invoked via an explicit
    venv python.exe path, not an activated shell) -- PATH is only a
    fallback."""
    venv_scripts_dir = Path(sys.executable).parent
    for candidate in (venv_scripts_dir / "podaac-data-downloader.exe", venv_scripts_dir / "podaac-data-downloader"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("podaac-data-downloader")
    if found:
        return found
    raise FileNotFoundError(
        "podaac-data-downloader not found -- install it with `pip install podaac-data-subscriber` "
        "in this venv first (see requirements.txt)"
    )


def _download_granules(config, raw_dir, force=False):
    """Always safe to call even when raw_dir already has granules --
    podaac-data-downloader queries the full CMR granule list for this
    date range/bbox itself and, by default (force=False), skips any file
    that already exists locally with a matching checksum, only fetching
    what's missing or doesn't match (e.g. a truncated prior download).
    force=True passes -f through, which re-downloads everything
    regardless of local state."""
    downloader = _find_downloader_executable()
    raw_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        downloader,
        "-c", COLLECTION,
        "-d", str(raw_dir),
        "-sd", f"{config['start_date']}T00:00:00Z",
        "-ed", f"{config['end_date']}T00:00:00Z",
    ]
    bbox = config.get("bbox")
    if bbox:
        cmd += ["-b", f"{bbox['lon_min']},{bbox['lat_min']},{bbox['lon_max']},{bbox['lat_max']}"]
    max_granules = config.get("max_granules")
    if max_granules:
        cmd += ["--limit", str(max_granules)]
    if force:
        cmd += ["-f"]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _granule_to_dataframe(path, max_qual_flag, open_ocean_flag_value):
    ds = xr.open_dataset(path)

    ssha = ds["ssha_karin_2"].values
    corr = ds["height_cor_xover"].values
    qual = ds["ssha_karin_2_qual"].values
    surface = ds["ancillary_surface_classification_flag"].values
    cross_track_distance = ds["cross_track_distance"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    # time is 1D (num_lines,) -- broadcast to (num_lines, num_pixels) to
    # match every other per-pixel variable before flattening.
    time = np.broadcast_to(ds["time"].values[:, None], ssha.shape)

    cycle = int(ds.attrs["cycle_number"])
    pass_number = int(ds.attrs["pass_number"])
    ds.close()

    # is_leftmost marks, per line (row of this 2D swath), the single pixel
    # closest to the left edge of the left swath (most negative
    # cross_track_distance -- see cross_track_distance's own comment attr:
    # negative = left side of swath) among pixels that will actually
    # survive the filtering below -- mirrors the dropna/quality_flag/
    # surface_flag conditions applied to df further down exactly, so an
    # is_leftmost=True point is always still present in the returned
    # dataframe, never filtered out afterwards. A line with zero surviving
    # pixels gets no is_leftmost point at all (nothing to flag). This lets
    # a caller later re-trace the swath's left-edge line without needing
    # the raw granule again.
    survives_filter = (
        ~np.isnan(ssha) & ~np.isnan(corr)
        & (qual == max_qual_flag) & (surface == open_ocean_flag_value)
    )
    masked_cross_track = np.where(survives_filter, cross_track_distance, np.inf)
    has_surviving_pixel = survives_filter.any(axis=1)
    leftmost_pixel_idx = masked_cross_track.argmin(axis=1)
    is_leftmost = np.zeros(ssha.shape, dtype=bool)
    is_leftmost[has_surviving_pixel, leftmost_pixel_idx[has_surviving_pixel]] = True

    df = pd.DataFrame({
        "lon": lon.ravel(),
        "lat": lat.ravel(),
        "time": time.ravel(),
        "ssha_karin_2": ssha.ravel(),
        "height_cor_xover": corr.ravel(),
        "quality_flag": qual.ravel(),
        "surface_flag": surface.ravel(),
        "is_leftmost": is_leftmost.ravel(),
    })
    df["cycle"] = cycle
    df["pass"] = pass_number

    df = df.dropna(subset=["ssha_karin_2", "height_cor_xover"])
    df = df[df["quality_flag"] == max_qual_flag]
    df = df[df["surface_flag"] == open_ocean_flag_value].copy()
    df["lon"] = np.where(df["lon"] > 180, df["lon"] - 360, df["lon"])
    df["swot_ssha"] = df["ssha_karin_2"] + df["height_cor_xover"]

    df["lon"] = df["lon"].round(6)
    df["lat"] = df["lat"].round(6)
    df["swot_ssha"] = df["swot_ssha"].round(4)

    return df[["lon", "lat", "time", "cycle", "pass", "swot_ssha", "quality_flag", "is_leftmost"]]


def main():
    parser = argparse.ArgumentParser(description="Download and canonicalize SWOT L2 KaRIn Expert granules into one Parquet table")
    parser.add_argument("--config", type=str, required=True, help="Path to a data_swot_l2 config YAML file")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download every granule even if it already exists locally with a matching checksum "
             "(passed through to podaac-data-downloader's own -f flag; without this, only missing/"
             "mismatched granules are (re-)fetched)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    product = config.get("product", "Expert")
    if product != "Expert":
        raise NotImplementedError(
            f"Only the Expert product is supported (height_cor_xover only exists there) -- "
            f"got product: {product!r}"
        )

    raw_dir = SCRIPT_DIR / "data" / "karin" / config["name"]
    print(f"Checking Expert granules for '{config['name']}' ({config['start_date']} to {config['end_date']})...")
    _download_granules(config, raw_dir, force=args.force)
    granule_paths = sorted(raw_dir.glob("SWOT_L2_LR_SSH_Expert_*.nc"))

    if not granule_paths:
        raise FileNotFoundError(
            f"No SWOT_L2_LR_SSH_Expert_*.nc granules found in {raw_dir} after download -- "
            f"check the date range/bbox actually covers a real SWOT pass"
        )

    max_qual_flag = config.get("max_qual_flag", 0)
    open_ocean_flag_value = config.get("open_ocean_flag_value", 0)
    print(f"Canonicalizing {len(granule_paths)} Expert granule(s)...")

    dfs = []
    for path in granule_paths:
        df = _granule_to_dataframe(path, max_qual_flag, open_ocean_flag_value)
        print(
            f"  {path.name}: {len(df)} point(s) kept "
            f"(quality_flag == {max_qual_flag}, surface_flag == {open_ocean_flag_value})"
        )
        dfs.append(df)

    points_df = pd.concat(dfs, ignore_index=True)

    drive_base, _ = get_drive_base_path()
    out_path = drive_base / "parquet files" / "swot_l2" / config["name"] / "points.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    points_df.to_parquet(out_path, index=False)

    print(f"Saved {len(points_df)} total point(s) to {out_path}")


if __name__ == "__main__":
    main()
