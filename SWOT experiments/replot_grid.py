"""
Regenerate final_mean.png / final_variance.png (the whole-globe grid
snapshot) for an already-completed experiment_5.py run, from its saved
checkpoints -- no Bayesian-optimization search or retraining needed.

Loads the 5 saved ensemble members from <results_dir>/checkpoints/, forward-
passes them over the same dense global lon/lat grid experiment_5.py itself
uses, and overwrites final_mean.png / final_variance.png in <results_dir>
with the imshow-style plots.

Usage:
    python replot_grid.py --results-dir results/exp_all_sats_1_day_random_shallow
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")  # must precede pyplot import -- see build_sliding_window_data.py's
                        # identical fix for the tkinter/Tcl crash this avoids when many
                        # figures get created+closed in a loop (--start-date/--end-date here,
                        # or any caller like build_validation_data.py that saves many plots)
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from DRF.models import DeepMaternRandomPhaseS2RFFNN
from model_io import load_checkpoint

# Matches the batch size used for grid/test predictions elsewhere in this
# codebase (e.g. train_model_process's grid_loader). Unbatched forward
# passes over the full grid worked fine at the old 512x256 resolution but
# OOM at 2048x1024 (~2.1M points in one shot) on an 8GB GPU.
_GRID_BATCH_SIZE = 8000

SCRIPT_DIR = Path(__file__).resolve().parent

# Kept in sync with build_experiment_data.py's own copy of this same
# constant -- the reference epoch for the unnormalized (time_znormalised:
# false) temporal representation, e.g. 2025-06-01 00:00 -> 9283.0 days.
TEMPORAL_REFERENCE_EPOCH = pd.Timestamp("2000-01-01")

# Kept in sync with experiment_5.py's copy of these same constants/helper --
# see that file for the full rationale (matplotlib's default figure sizing
# silently resamples the grid to fit a fixed pixel budget regardless of
# _NUM_LONGS/_NUM_LATS, unless the map gets its own explicitly-sized axes).
_NUM_LONGS = 2880  # 0.125 deg/pixel: 360 / 0.125
_NUM_LATS = 1440   # 0.125 deg/pixel: 180 / 0.125

# Cell-CENTER coordinates of the first/last pixel, e.g. -179.9375/179.9375
# for _NUM_LONGS=2880 -- NOT -180/180. torch.linspace(-180, 180, _NUM_LONGS)
# (this file's/experiment_5.py's original construction) places a point
# exactly ON each edge instead, which for _NUM_LONGS points spread over 360
# degrees inclusive of both endpoints gives spacing 360/(_NUM_LONGS-1) --
# subtly wider than the true 360/_NUM_LONGS pixel width, and phase-shifted
# by about half a pixel from DUACS's own real grid (verified directly
# against its zarr coordinates: exactly 0.125 deg spacing, first point
# -179.9375, i.e. cell-centered). That mismatch meant a DRF "nearest pixel"
# lookup and a DUACS "nearest pixel" lookup could disagree about which grid
# cell a given point belongs to -- caught via two SWOT points close enough
# to share one DUACS pixel but split across two different DRF ones.
# Building the grid as pixel CENTERS (this formula) instead of edge-to-edge
# makes DRF's grid exactly match DUACS's.
_LON_PIXEL_WIDTH = 360 / _NUM_LONGS
_LAT_PIXEL_WIDTH = 180 / _NUM_LATS
_GRID_LON_MIN = -180 + _LON_PIXEL_WIDTH / 2
_GRID_LON_MAX = 180 - _LON_PIXEL_WIDTH / 2
_GRID_LAT_MIN = -90 + _LAT_PIXEL_WIDTH / 2
_GRID_LAT_MAX = 90 - _LAT_PIXEL_WIDTH / 2

_GRID_DPI = 256
_COLORBAR_HEIGHT_PX = 256  # legend space only, not pixel-critical
_BORDER_PX = 32  # small uniform white margin around the whole saved image

# Gulf Stream bounding box for the --gulf zoom crop below. Lon/lat MIN
# match build_gulf_stream_mask.py's GULF_STREAM_LON_MIN/LAT_MIN exactly;
# MAX is deliberately wider here (that mask's box is tuned tight around the
# high-eddy-activity current itself, whereas this crop is a visualization
# and benefits from showing more surrounding context). All four edges are
# still exact multiples of the 0.125deg pixel width (82/0.125=656,
# 20/0.125=160, 25/0.125=200, 50/0.125=400, all integers), i.e. the box
# falls exactly on pixel EDGES on this grid -- so a "pixels whose center
# lies inside the box" crop reproduces these edges exactly, zero rounding
# drift.
_GULF_LON_MIN = -82.0
_GULF_LON_MAX = -20.0
_GULF_LAT_MIN = 25.0
_GULF_LAT_MAX = 50.0


def _gulf_pixel_bounds():
    """Pixel index bounds (as half-open ranges) of the Gulf Stream box on
    the canonical _NUM_LONGS x _NUM_LATS grid, plus the crop's true outer
    edges in degrees (derived from the included pixels' own index spacing,
    not the raw box constants, avoiding any float drift -- same pattern as
    plot_swot_track_overlays_v2.py's _load_grid16)."""
    grid_lons_deg = torch.linspace(_GRID_LON_MIN, _GRID_LON_MAX, _NUM_LONGS)
    grid_lats_deg = torch.linspace(_GRID_LAT_MIN, _GRID_LAT_MAX, _NUM_LATS)
    lon_idx = torch.nonzero((grid_lons_deg >= _GULF_LON_MIN) & (grid_lons_deg <= _GULF_LON_MAX)).flatten()
    lat_idx = torch.nonzero((grid_lats_deg >= _GULF_LAT_MIN) & (grid_lats_deg <= _GULF_LAT_MAX)).flatten()
    lon_i0, lon_i1 = int(lon_idx[0]), int(lon_idx[-1]) + 1
    lat_i0, lat_i1 = int(lat_idx[0]), int(lat_idx[-1]) + 1
    edge_lon_min = _GRID_LON_MIN + lon_i0 * _LON_PIXEL_WIDTH - _LON_PIXEL_WIDTH / 2
    edge_lon_max = _GRID_LON_MIN + (lon_i1 - 1) * _LON_PIXEL_WIDTH + _LON_PIXEL_WIDTH / 2
    edge_lat_min = _GRID_LAT_MIN + lat_i0 * _LAT_PIXEL_WIDTH - _LAT_PIXEL_WIDTH / 2
    edge_lat_max = _GRID_LAT_MIN + (lat_i1 - 1) * _LAT_PIXEL_WIDTH + _LAT_PIXEL_WIDTH / 2
    return lon_i0, lon_i1, lat_i0, lat_i1, edge_lon_min, edge_lon_max, edge_lat_min, edge_lat_max


(
    _GULF_LON_I0, _GULF_LON_I1, _GULF_LAT_I0, _GULF_LAT_I1,
    _GULF_EDGE_LON_MIN, _GULF_EDGE_LON_MAX, _GULF_EDGE_LAT_MIN, _GULF_EDGE_LAT_MAX,
) = _gulf_pixel_bounds()

# Gulf crop map width is pinned to _NUM_LONGS (same as the whole-globe
# plots, so gulf_*.png reads as "the same big picture, zoomed in"); map
# HEIGHT is instead derived from the crop's own lon/lat aspect ratio at
# that fixed width. Reusing _NUM_LATS (the whole globe's 180deg-tall
# aspect) here would letterbox the much shorter Gulf box, leaving dead
# white space above/below -- a PlateCarree GeoAxes preserves 1 degree lon
# == 1 degree lat on screen regardless of the axes box's own pixel shape,
# so the box's pixel aspect has to match the data's degree aspect for the
# map to fill it exactly.
_GULF_MAP_WIDTH_PX = _NUM_LONGS
_GULF_MAP_HEIGHT_PX = (
    _GULF_MAP_WIDTH_PX * (_GULF_EDGE_LAT_MAX - _GULF_EDGE_LAT_MIN) / (_GULF_EDGE_LON_MAX - _GULF_EDGE_LON_MIN)
)


def _save_global_grid_plot(
    data_np, cmap, vmin, vmax, colorbar_label, save_path, mask_land=False,
    num_lons=_NUM_LONGS, num_lats=_NUM_LATS,
    lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0,
):
    """Saves a grid snapshot with the map rendered at exactly num_lons x
    num_lats pixels -- no interpolation/resampling -- inset by a uniform
    _BORDER_PX white margin on all four sides. Defaults cover the whole
    globe (matching experiment_5.py's copy of this function, which has the
    full rationale); passing a smaller num_lons/num_lats + a narrower
    lon/lat range (e.g. _save_gulf_crop_plot below) renders a pixel-exact
    zoomed crop instead, using the same fig.add_axes() pixel-fraction
    technique so it never gets resampled either."""
    total_width_px = num_lons + 2 * _BORDER_PX
    total_height_px = num_lats + _COLORBAR_HEIGHT_PX + 2 * _BORDER_PX
    fig = plt.figure(figsize=(total_width_px / _GRID_DPI, total_height_px / _GRID_DPI), dpi=_GRID_DPI)
    map_left = _BORDER_PX / total_width_px
    map_width = num_lons / total_width_px
    map_bottom = (_BORDER_PX + _COLORBAR_HEIGHT_PX) / total_height_px
    map_height = num_lats / total_height_px
    ax = fig.add_axes(
        [map_left, map_bottom, map_width, map_height],
        projection=ccrs.PlateCarree(central_longitude=0),
    )
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    im = ax.imshow(
        data_np,
        origin="lower",
        cmap=cmap,
        extent=[lon_min, lon_max, lat_min, lat_max],
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
        interpolation="none",
    )
    if mask_land:
        # The model predicts a value everywhere (it has no notion of land vs
        # ocean), so land pixels are real predictions, not missing data --
        # this is a purely visual overlay (solid fill drawn on top of the
        # data), not a NaN mask of data_np itself. cfeature.LAND is a
        # Natural Earth polygon dataset cartopy already ships, so this needs
        # no new dependency and no raster land/sea mask. zorder=1 (below the
        # coastline/border zorders below) -- left at the cartopy/matplotlib
        # default, LAND and the outlines land at the same effective zorder,
        # and LAND (added first) can still end up painted last and hide the
        # outlines underneath its opaque fill.
        ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=1)
    # Google-Maps-style outlines: a bolder solid coastline, dotted (not
    # solid) country borders. Explicit zorder=2 keeps both above the LAND
    # mask fill regardless of add-order/library-default zorder ties.
    ax.coastlines(linewidth=1.0, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=1.0, zorder=2)
    cax_left = map_left + 0.15 * map_width
    cax_width = 0.7 * map_width
    cax_bottom = _BORDER_PX / total_height_px + 0.35 * (_COLORBAR_HEIGHT_PX / total_height_px)
    cax_height = 0.3 * (_COLORBAR_HEIGHT_PX / total_height_px)
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=colorbar_label)
    fig.savefig(save_path, dpi=_GRID_DPI)
    plt.close(fig)


def to_float(x):
    return x.item() if torch.is_tensor(x) else x


def build_grid_inputs(date, temporal_mean, temporal_std, temporal_znormalised):
    """Returns (grid_spatial_X, grid_temporal_X) for a dense whole-globe
    grid at the given date -- the same construction the --date CLI branch
    below uses, factored out so other scripts (e.g. an evaluation pipeline
    comparing DRF against SWOT/DUACS at specific dates) can request a grid
    at an arbitrary date without duplicating the temporal-normalization
    logic. date may be a string or pd.Timestamp."""
    grid_lons_deg = torch.linspace(_GRID_LON_MIN, _GRID_LON_MAX, _NUM_LONGS)
    grid_lats_deg = torch.linspace(_GRID_LAT_MIN, _GRID_LAT_MAX, _NUM_LATS)
    grid_lon_grid, grid_lat_grid = torch.meshgrid(grid_lons_deg, grid_lats_deg, indexing="ij")
    grid_spatial_X = torch.stack(
        [torch.deg2rad(grid_lon_grid.reshape(-1)), torch.deg2rad(grid_lat_grid.reshape(-1))], dim=1,
    )
    target_timestamp = pd.Timestamp(date)
    if temporal_znormalised:
        target_raw_seconds = target_timestamp.timestamp()
        target_normalized = (target_raw_seconds - temporal_mean) / (temporal_std + 1e-8)
    else:
        target_normalized = (target_timestamp - TEMPORAL_REFERENCE_EPOCH) / pd.Timedelta(days=1)
    grid_temporal_X = torch.full((grid_spatial_X.shape[0], 1), target_normalized, dtype=torch.float32)
    return grid_spatial_X, grid_temporal_X


def predict_grid(models, device, grid_spatial_X, grid_temporal_X):
    """Forward-passes an ensemble over a grid, returning (mean, var) each
    reshaped to (_NUM_LONGS, _NUM_LATS) -- the actual (2880, 1440) values,
    not yet plotted/saved. Pulled out of _predict_and_save_grid so a caller
    that just wants the array (e.g. to diff against DUACS, or to sample at
    SWOT points) doesn't need to go through plotting/saving at all."""
    grid_loader = DataLoader(
        TensorDataset(grid_spatial_X, grid_temporal_X),
        batch_size=_GRID_BATCH_SIZE,
        shuffle=False,
    )
    with torch.no_grad():
        grid_per_model_preds_list = []
        for model in models:
            batch_preds = []
            for batch_spatial, batch_temporal in grid_loader:
                batch_preds.append(
                    model(batch_spatial.to(device), batch_temporal.to(device)).cpu()
                )
            grid_per_model_preds_list.append(torch.cat(batch_preds, dim=0))
        grid_per_model_preds = torch.stack(grid_per_model_preds_list)  # [num_models, N_grid, 1]

    grid_mean_pred = grid_per_model_preds.mean(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    grid_var_pred = grid_per_model_preds.var(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    return grid_mean_pred, grid_var_pred


def _save_gulf_crop_plot(grid_pred, cmap, vmin, vmax, colorbar_label, save_path, mask_land):
    """Crop of the Gulf Stream bounding box (_GULF_LON_I0:_GULF_LON_I1,
    _GULF_LAT_I0:_GULF_LAT_I1) out of a full-globe grid_pred tensor shaped
    (_NUM_LONGS, _NUM_LATS) -- sliced by index (same technique as
    plot_swot_track_overlays_v2.py's _load_grid16), so every rendered value
    is a genuine, unaltered grid cell, never blended/resampled.

    Map width (_GULF_MAP_WIDTH_PX) matches the whole-globe plots exactly,
    so gulf_*.png reads as "the same big picture, zoomed in", with the same
    colorbar/border/font size. Map height (_GULF_MAP_HEIGHT_PX) is instead
    sized to the crop's own lon/lat aspect ratio at that width -- reusing
    the globe's height here would letterbox the much shorter Gulf box with
    dead white space above/below, so the overall saved image ends up
    shorter than the whole-globe one, not padded to match it.
    interpolation stays "none", so each source cell still renders as one
    sharp block (just a bigger one), not a smoothed reprojection."""
    crop = grid_pred[_GULF_LON_I0:_GULF_LON_I1, _GULF_LAT_I0:_GULF_LAT_I1]
    _save_global_grid_plot(
        crop.T.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label=colorbar_label, save_path=save_path,
        mask_land=mask_land, num_lons=_GULF_MAP_WIDTH_PX, num_lats=_GULF_MAP_HEIGHT_PX,
        lon_min=_GULF_EDGE_LON_MIN, lon_max=_GULF_EDGE_LON_MAX, lat_min=_GULF_EDGE_LAT_MIN, lat_max=_GULF_EDGE_LAT_MAX,
    )


def _save_mean_variance_plots(grid_mean_pred, grid_var_pred, date_label, mean_path, variance_path, mask_land, gulf):
    """Saves the full-globe mean/variance snapshots and, if gulf, a second
    pixel-exact pair zoomed into the Gulf Stream box alongside them, named
    gulf_<original file name> (e.g. gulf_final_mean.png) -- reused by every
    --date/--start-date/default branch below so gulf support only needs
    implementing once. date_label is the exact string to show in the
    colorbar -- None keeps the label ambiguous (matches experiment_5.py's
    own final_mean.png/final_variance.png, which likewise never claims a
    specific timestamp, only "the training set's mean", to avoid the two
    ever silently disagreeing); a real string (from --date/--start-date)
    states plainly which timestamp this particular snapshot was predicted at."""
    label_suffix = f" -- {date_label}" if date_label else ""
    _save_global_grid_plot(
        grid_mean_pred.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
        colorbar_label=f"DRF Predicted SLA (m){label_suffix}", save_path=mean_path,
        mask_land=mask_land,
    )
    _save_global_grid_plot(
        grid_var_pred.T.numpy(), cmap="viridis", vmin=0, vmax=0.1,
        colorbar_label=f"DRF Variance{label_suffix}", save_path=variance_path,
        mask_land=mask_land,
    )
    if gulf:
        _save_gulf_crop_plot(
            grid_mean_pred, cmap="coolwarm", vmin=-0.25, vmax=0.25,
            colorbar_label=f"DRF Predicted SLA (m){label_suffix} -- Gulf Stream",
            save_path=mean_path.with_name(f"gulf_{mean_path.name}"), mask_land=mask_land,
        )
        _save_gulf_crop_plot(
            grid_var_pred, cmap="viridis", vmin=0, vmax=0.1,
            colorbar_label=f"DRF Variance{label_suffix} -- Gulf Stream",
            save_path=variance_path.with_name(f"gulf_{variance_path.name}"), mask_land=mask_land,
        )


def _predict_and_save_grid(final_models, grid_spatial_X, grid_temporal_X, device, date_label, mean_path, variance_path, mask_land, gulf=False):
    """Predicts final_models over the given grid, then saves the mean/
    variance snapshots (plus Gulf Stream crops if gulf) via
    _save_mean_variance_plots."""
    grid_mean_pred, grid_var_pred = predict_grid(final_models, device, grid_spatial_X, grid_temporal_X)
    _save_mean_variance_plots(grid_mean_pred, grid_var_pred, date_label, mean_path, variance_path, mask_land, gulf)


def load_ensemble(checkpoints_dir, device):
    checkpoint_paths = sorted(checkpoints_dir.glob("model_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir} -- this script only "
            f"works on a results folder produced by experiment_5.py, which "
            f"is the only script in this codebase that saves model weights."
        )

    models = []
    temporal_mean = None
    temporal_std = None
    temporal_znormalised = None
    for path in checkpoint_paths:
        checkpoint = load_checkpoint(path)
        if temporal_mean is None:
            # Raw mean/std the training split's temporal column was
            # summarized with (saved by build_experiment_data.py) -- in
            # seconds-since-1970 if z-scored, or days-since-2000 if not (see
            # temporal_znormalised). Needed to turn a user-supplied --date
            # into whatever this model actually expects. Every ensemble
            # member was trained on the same split, so any one checkpoint's
            # stats are representative. .get(..., True) defaults to the
            # z-scored assumption for checkpoints saved before this flag
            # existed -- the only behaviour there ever was until now.
            temporal_mean = checkpoint["normalization_stats"]["temporal_mean"].item()
            temporal_std = checkpoint["normalization_stats"]["temporal_std"].item()
            temporal_znormalised = checkpoint["normalization_stats"].get("temporal_znormalised", True)
        (
            spatial_lengthscale,
            temporal_lengthscale,
            amplitude,
            lengthscale2,
            amplitude2,
        ) = checkpoint["hyperparameters"]

        # MaternRandomPhaseS2RFFLayer's random spherical basis (noise/levels
        # in RandomPhaseFeatureMap) is sampled fresh at construction time and
        # is NOT part of state_dict (RandomPhaseFeatureMap isn't an
        # nn.Module, so nothing in it gets registered as a parameter/buffer).
        # Re-seeding with the checkpoint's own saved seed before
        # constructing the model reproduces the exact basis the saved
        # output_layer weights were actually trained against -- without
        # this, each reconstruction gets an unrelated random basis and the
        # loaded weights are combined with the wrong features.
        torch.manual_seed(checkpoint["seed"])
        model = DeepMaternRandomPhaseS2RFFNN(
            **checkpoint["model_config"],
            spatial_lengthscale=to_float(spatial_lengthscale),
            temporal_lengthscale=to_float(temporal_lengthscale),
            amplitude=to_float(amplitude),
            lengthscale2=to_float(lengthscale2),
            amplitude2=to_float(amplitude2),
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} model(s) from {checkpoints_dir}")
    return models, temporal_mean, temporal_std, temporal_znormalised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate whole-globe grid plots from a saved exp5 ensemble"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to a results/<config-name> folder containing checkpoints/",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mask-land", action="store_true",
        help="Cover land with a solid white overlay (Natural Earth polygons via cartopy) -- purely visual, the model still predicts a value there.",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help=(
            "Predict at this exact timestamp instead of the training set's "
            "mean, e.g. '2025-06-01 00:00'. Writes SEPARATE "
            "final_mean_<date>.png/final_variance_<date>.png files -- does "
            "NOT touch/regenerate the existing final_mean.png/"
            "final_variance.png (those stay as experiment_5.py originally "
            "produced them). A date far outside the model's training range "
            "is extrapolation. Mutually exclusive with --start-date/--end-date."
        ),
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Predict one grid per day from this date through --end-date (inclusive), e.g. '2025-07-01'.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End of the --start-date range (inclusive).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="With --start-date/--end-date, rebuild every day even if already saved (default: skip days already saved).",
    )
    parser.add_argument(
        "--gulf", action="store_true",
        help=(
            "Also save a pixel-exact crop zoomed into the Gulf Stream region "
            "(same box as build_gulf_stream_mask.py) alongside every normal "
            "mean/variance image, named gulf_<original file name> (e.g. "
            "final_mean.png -> gulf_final_mean.png). No extra model inference -- "
            "reuses the same predicted grid, just cropped and re-rendered. "
            "Combines with --date/--start-date/--end-date/--mask-land."
        ),
    )
    args = parser.parse_args()

    if args.date is not None and (args.start_date is not None or args.end_date is not None):
        parser.error("--date is mutually exclusive with --start-date/--end-date")
    if bool(args.start_date) != bool(args.end_date):
        parser.error("--start-date and --end-date must be given together")

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir
    device = torch.device(args.device)

    final_models, temporal_mean, temporal_std, temporal_znormalised = load_ensemble(
        results_dir / "checkpoints", device
    )

    if args.start_date is not None:
        # One grid per day over a range -- e.g. so a fixed-window run can be
        # evaluated against a whole validation period without retraining.
        # Loads the ensemble once (already done above), then loops days,
        # saving both the raw array (.pt, for build_validation_data.py to
        # read directly -- no re-inference needed there) and the picture
        # (.png) per day into results_dir/grids/, skipping a day already
        # saved unless --force (same resumability as everything else in
        # this pipeline).
        grids_dir = results_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(args.start_date, args.end_date, freq="1D")]
        print(f"Building {len(target_dates)} whole-globe grid snapshot(s) from {args.start_date} to {args.end_date}...")

        for i, date_str in enumerate(target_dates, 1):
            mean_pt_path = grids_dir / f"{date_str}_mean.pt"
            var_pt_path = grids_dir / f"{date_str}_variance.pt"
            if not args.force and mean_pt_path.exists() and var_pt_path.exists():
                print(f"[{i}/{len(target_dates)}] {date_str}: already saved, skipping.")
                continue

            print(f"[{i}/{len(target_dates)}] {date_str}: predicting grid...")
            grid_spatial_X, grid_temporal_X = build_grid_inputs(date_str, temporal_mean, temporal_std, temporal_znormalised)
            grid_mean, grid_var = predict_grid(final_models, device, grid_spatial_X, grid_temporal_X)

            torch.save(grid_mean, mean_pt_path)
            torch.save(grid_var, var_pt_path)
            _save_mean_variance_plots(
                grid_mean, grid_var, date_str,
                grids_dir / f"{date_str}_mean.png", grids_dir / f"{date_str}_variance.png",
                args.mask_land, args.gulf,
            )
        print(f"Done. Grids saved in {grids_dir}")
    elif args.date is None:
        print("Building global grid for whole-globe snapshot...")
        grid_lons_deg = torch.linspace(_GRID_LON_MIN, _GRID_LON_MAX, _NUM_LONGS)
        grid_lats_deg = torch.linspace(_GRID_LAT_MIN, _GRID_LAT_MAX, _NUM_LATS)
        grid_lon_grid, grid_lat_grid = torch.meshgrid(grid_lons_deg, grid_lats_deg, indexing="ij")
        grid_spatial_X = torch.stack(
            [torch.deg2rad(grid_lon_grid.reshape(-1)), torch.deg2rad(grid_lat_grid.reshape(-1))],
            dim=1,
        )
        # Default timestamp: the training set's mean. Normalized time = 0.0
        # IS that mean when the data was z-scored; when time_znormalised was
        # False at build_experiment_data.py time, temporal_X is raw
        # days-since-2000, so 0.0 would mean the year 2000 instead -- the
        # actual stored mean has to be fed in (mirrors experiment_5.py's
        # own copy of this same branch).
        if temporal_znormalised:
            grid_temporal_X = torch.zeros(grid_spatial_X.shape[0], 1)
        else:
            grid_temporal_X = torch.full((grid_spatial_X.shape[0], 1), temporal_mean, dtype=torch.float32)

        _predict_and_save_grid(
            final_models, grid_spatial_X, grid_temporal_X, device, None,
            results_dir / "final_mean.png", results_dir / "final_variance.png", args.mask_land,
            gulf=args.gulf,
        )
        print("Global grid predictions computed.")
        print(f"Regenerated final_mean.png and final_variance.png in {results_dir}")
    else:
        try:
            target_timestamp = pd.Timestamp(args.date)
        except ValueError as e:
            raise ValueError(
                f"Could not parse --date '{args.date}' -- expected a format "
                f"like '2025-06-01 00:00'"
            ) from e
        # Slash date / colon time, matching build_l4_data.py's L4 overview
        # plot labels (e.g. "DUACS SLA (m) -- 2025/06/01 00:00") -- derived
        # from the parsed timestamp (not a raw string replace on args.date)
        # so it's consistently formatted regardless of how --date was typed.
        date_label = target_timestamp.strftime("%Y/%m/%d %H:%M")

        print(f"Building global grid for whole-globe snapshot at {args.date}...")
        grid_spatial_X, grid_temporal_X = build_grid_inputs(
            target_timestamp, temporal_mean, temporal_std, temporal_znormalised
        )

        # Filesystem-safe (Windows disallows ":") while keeping the date
        # readable, e.g. "2025-06-01 00:00" -> "2025-06-01_00-00". Derived
        # from the same parsed target_timestamp as date_label, so the
        # filename is consistently zero-padded regardless of how --date was
        # typed (e.g. "2025-6-1 0:0" still produces "2025-06-01_00-00").
        date_tag = target_timestamp.strftime("%Y-%m-%d_%H-%M")
        mean_path = results_dir / f"final_mean_{date_tag}.png"
        variance_path = results_dir / f"final_variance_{date_tag}.png"
        _predict_and_save_grid(
            final_models, grid_spatial_X, grid_temporal_X, device, date_label,
            mean_path, variance_path, args.mask_land,
            gulf=args.gulf,
        )
        print("Global grid predictions computed.")
        print(f"Saved {mean_path.name} and {variance_path.name} in {results_dir}")