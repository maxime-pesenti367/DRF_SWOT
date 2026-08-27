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
_GRID_DPI = 256
_COLORBAR_HEIGHT_PX = 256  # legend space only, not pixel-critical
_BORDER_PX = 32  # small uniform white margin around the whole saved image

_TOTAL_WIDTH_PX = _NUM_LONGS + 2 * _BORDER_PX
_TOTAL_HEIGHT_PX = _NUM_LATS + _COLORBAR_HEIGHT_PX + 2 * _BORDER_PX
_FIG_WIDTH_IN = _TOTAL_WIDTH_PX / _GRID_DPI
_FIG_HEIGHT_IN = _TOTAL_HEIGHT_PX / _GRID_DPI


def _save_global_grid_plot(data_np, cmap, vmin, vmax, colorbar_label, save_path, mask_land=False):
    """Saves a whole-globe grid snapshot with the map rendered at exactly
    _NUM_LONGS x _NUM_LATS pixels -- no interpolation/resampling -- inset by
    a uniform _BORDER_PX white margin on all four sides. See experiment_5.py's
    copy of this function for the full rationale."""
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
        extent=[-180, 180, -90, 90],
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
    cax_bottom = _BORDER_PX / _TOTAL_HEIGHT_PX + 0.35 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
    cax_height = 0.3 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
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
    grid_lons_deg = torch.linspace(-180, 180, _NUM_LONGS)
    grid_lats_deg = torch.linspace(-90, 90, _NUM_LATS)
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


def _predict_and_save_grid(final_models, grid_spatial_X, grid_temporal_X, device, date_label, mean_path, variance_path, mask_land):
    """Predicts final_models over the given grid and saves the mean/
    variance snapshots. date_label is the exact string to show in the
    colorbar -- None keeps the label ambiguous (matches experiment_5.py's
    own final_mean.png/final_variance.png, which likewise never claims a
    specific timestamp, only "the training set's mean", to avoid the two
    ever silently disagreeing); a real string (from --date) states plainly
    which timestamp this particular snapshot was predicted at."""
    grid_mean_pred, grid_var_pred = predict_grid(final_models, device, grid_spatial_X, grid_temporal_X)

    label_suffix = f" -- {date_label}" if date_label else ""
    _save_global_grid_plot(
        grid_mean_pred.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
        colorbar_label=f"DRF Predicted SLA (m){label_suffix}", save_path=mean_path,
        mask_land=mask_land,
    )
    _save_global_grid_plot(
        grid_var_pred.T.numpy(), cmap="viridis", vmin=0, vmax=0.2,
        colorbar_label=f"DRF Variance{label_suffix}", save_path=variance_path,
        mask_land=mask_land,
    )


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
            "is extrapolation."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir
    device = torch.device(args.device)

    final_models, temporal_mean, temporal_std, temporal_znormalised = load_ensemble(
        results_dir / "checkpoints", device
    )

    if args.date is None:
        print("Building global grid for whole-globe snapshot...")
        grid_lons_deg = torch.linspace(-180, 180, _NUM_LONGS)
        grid_lats_deg = torch.linspace(-90, 90, _NUM_LATS)
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
        )
        print("Global grid predictions computed.")
        print(f"Saved {mean_path.name} and {variance_path.name} in {results_dir}")