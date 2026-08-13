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
import torch

from DRF.models import DeepMaternRandomPhaseS2RFFNN
from model_io import load_checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent

_NUM_LONGS = 512
_NUM_LATS = 256


def to_float(x):
    return x.item() if torch.is_tensor(x) else x


def load_ensemble(checkpoints_dir, device):
    checkpoint_paths = sorted(checkpoints_dir.glob("model_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir} -- this script only "
            f"works on a results folder produced by experiment_5.py, which "
            f"is the only script in this codebase that saves model weights."
        )

    models = []
    for path in checkpoint_paths:
        checkpoint = load_checkpoint(path)
        (
            spatial_lengthscale,
            temporal_lengthscale,
            amplitude,
            lengthscale2,
            amplitude2,
        ) = checkpoint["hyperparameters"]

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
    return models


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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir
    device = torch.device(args.device)

    final_models = load_ensemble(results_dir / "checkpoints", device)

    print("Building global grid for whole-globe snapshot...")
    grid_lons_deg = torch.linspace(-180, 180, _NUM_LONGS)
    grid_lats_deg = torch.linspace(-90, 90, _NUM_LATS)
    grid_lon_grid, grid_lat_grid = torch.meshgrid(grid_lons_deg, grid_lats_deg, indexing="ij")
    grid_spatial_X = torch.stack(
        [torch.deg2rad(grid_lon_grid.reshape(-1)), torch.deg2rad(grid_lat_grid.reshape(-1))],
        dim=1,
    ).to(device)
    # Normalized time = 0.0 -> the training set's mean timestamp.
    grid_temporal_X = torch.zeros(grid_spatial_X.shape[0], 1, device=device)

    with torch.no_grad():
        grid_per_model_preds = torch.stack(
            [model(grid_spatial_X, grid_temporal_X).cpu() for model in final_models]
        )  # [num_models, N_grid, 1]

    grid_mean_pred = grid_per_model_preds.mean(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    grid_var_pred = grid_per_model_preds.var(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    print("Global grid predictions computed.")

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})
    mean_plot = ax.imshow(
        grid_mean_pred.T.numpy(),
        origin="lower",
        cmap="coolwarm",
        extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(),
        vmin=-0.25,
        vmax=0.25,
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plt.colorbar(mean_plot, ax=ax, orientation="horizontal", pad=0.05, label="Predicted SLA (m)")
    plt.savefig(results_dir / "final_mean.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})
    var_plot = ax.imshow(
        grid_var_pred.T.numpy(),
        origin="lower",
        cmap="viridis",
        extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=0.2,
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plt.colorbar(var_plot, ax=ax, orientation="horizontal", pad=0.05, label="Variance")
    plt.savefig(results_dir / "final_variance.png", dpi=300)
    plt.close(fig)

    print(f"Regenerated final_mean.png and final_variance.png in {results_dir}")