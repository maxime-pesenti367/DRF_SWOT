"""
Canonical spherical-DRF training/eval script — the intended replacement for
experiment_3.py / experiment_4.py going forward. See
exp5_implementation_plan.md (and exp3_exp4_findings_and_exp5_plan.md before
it) for the full reasoning behind every design choice here.

Usage:
    python experiment_5.py --config configs/exp5/exp_all_sats_1_day_random_shallow.yaml

Consumes tensors already split and normalized by build_experiment_data.py —
this script has zero knowledge of split strategy and does not re-split
anything.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, TensorDataset

from DRF.models import DeepMaternRandomPhaseS2RFFNN
from DRF.spherical_uq_methods import SphericalBayesianOptimizer
from copernicus_pipeline import get_drive_base_path
from model_io import save_checkpoint

SCRIPT_DIR = Path(__file__).resolve().parent


def compute_rmse(mu, y):
    return torch.sqrt(torch.mean((mu - y) ** 2)).item()


def compute_nlpd(mu, var, y):
    loss_fn = nn.GaussianNLLLoss(full=True, eps=1e-6, reduction="mean")
    return loss_fn(mu, y, var).item()


def compute_crps(mu, var, y):
    """Closed-form CRPS for a Gaussian predictive distribution
    (Gneiting & Raftery, 2007)."""
    sigma = var.clamp(min=1e-12).sqrt()
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    pdf_z = torch.exp(normal.log_prob(z))
    cdf_z = normal.cdf(z)
    crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1.0 / torch.sqrt(torch.tensor(torch.pi)))
    return crps.mean().item()


def train_final_model(
    model_class,
    train_data,
    num_layers,
    spatial_input_dim,
    temporal_input_dim,
    hidden_dim,
    bottleneck_dim,
    output_dim,
    nu,
    lon_lat_inputs,
    combine_type,
    hyperparams,
    seed,
    device,
    n_epochs,
    batch_size,
):
    """
    Trains one model to completion and returns the model object itself (not
    just its predictions), so its weights can be saved afterwards.

    Deliberately NOT routed through DRF.spherical_uq_methods.train_model_process
    — that function is reused as-is for the Bayesian-optimization search
    phase below, but it only ever returns predictions, never the trained
    model, and extending its return contract would require also touching
    experiment_3.py/experiment_4.py's existing calls to it, which is out of
    scope here. This is a small, deliberate duplication of its training loop
    for the one new thing it doesn't support: handing back a model whose
    weights can actually be persisted.
    """
    torch.manual_seed(seed)
    spatial_lengthscale, temporal_lengthscale, amplitude, lengthscale2, amplitude2 = hyperparams

    def to_float(x):
        return x.item() if torch.is_tensor(x) else x

    model = model_class(
        num_layers=num_layers,
        spatial_input_dim=spatial_input_dim,
        temporal_input_dim=temporal_input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        spatial_lengthscale=to_float(spatial_lengthscale),
        temporal_lengthscale=to_float(temporal_lengthscale),
        nu=nu,
        amplitude=to_float(amplitude),
        lengthscale2=to_float(lengthscale2),
        amplitude2=to_float(amplitude2),
        lon_lat_inputs=lon_lat_inputs,
        combine_type=combine_type,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.HuberLoss(delta=0.1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(n_epochs):
        for batch in train_loader:
            batch_spatial, batch_temporal, batch_values = batch
            batch_spatial = batch_spatial.to(device)
            batch_temporal = batch_temporal.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()
            outputs = model(batch_spatial, batch_temporal)
            loss = criterion(outputs, batch_values)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a spherical DRF model (exp5)")
    parser.add_argument("--config", type=str, required=True, help="Path to a model/training config YAML file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])

    # Resolved the same way build_experiment_data.py saved it -- via
    # get_drive_base_path(), not a hardcoded absolute path in the config.
    # A path like 'G:/My Drive/...' only means anything on Windows; baking
    # it into a config that's committed to git and run on multiple machines
    # (e.g. this local machine and a remote Linux GPU cluster) breaks the
    # moment it runs somewhere without that drive letter.
    drive_base, _ = get_drive_base_path()
    tensor_path = (
        drive_base / "pytorch tensors" / config["data"]["experiment_name"]
        / f"{config['data']['split_name']}.pt"
    )
    print(f"Loading pre-split, pre-normalized tensors from {tensor_path}...")
    if not tensor_path.exists():
        raise FileNotFoundError(
            f"Tensor file not found at: {tensor_path}\n"
            f"Likely causes:\n"
            f"  - build_experiment_data.py hasn't been run for "
            f"'{config['data']['experiment_name']}' yet\n"
            f"  - it was built on a different machine and needs to be "
            f"copied here (e.g. via scp) into the path above\n"
            f"  - on this machine, get_drive_base_path() has no real Drive "
            f"mount and is falling back to a local folder (override with "
            f"the SWOT_DATA_DIR env var if the file lives somewhere else)"
        )
    data = torch.load(tensor_path, weights_only=False, map_location="cpu")

    spatial_X_train = data["spatial_X_train"]
    temporal_X_train = data["temporal_X_train"]
    y_train = data["y_train"].unsqueeze(-1)

    spatial_X_val = data["spatial_X_val"]
    temporal_X_val = data["temporal_X_val"]
    y_val = data["y_val"].unsqueeze(-1)

    spatial_X_test = data["spatial_X_test"]
    temporal_X_test = data["temporal_X_test"]
    y_test = data["y_test"].unsqueeze(-1)

    train_dataset = TensorDataset(spatial_X_train, temporal_X_train, y_train)
    val_dataset = TensorDataset(spatial_X_val, temporal_X_val, y_val)
    # Unlabeled — only used by the BO search phase's internal prediction
    # step below, which this script otherwise ignores in favour of scoring
    # the final retrained ensemble against the real labeled test set later.
    test_dataset_unlabeled = TensorDataset(spatial_X_test, temporal_X_test)

    # Grid spacing for the spatial functional-regularization term.
    lons = spatial_X_train[:, 0]
    lats = spatial_X_train[:, 1]
    unique_lons = torch.unique(lons)
    unique_lats = torch.unique(lats)
    d_phi = torch.abs(unique_lons[1] - unique_lons[0]).item() if len(unique_lons) > 1 else 0.0
    d_theta = torch.abs(unique_lats[1] - unique_lats[0]).item() if len(unique_lats) > 1 else 0.0

    nu = config["model"]["kwargs"].get("nu", 1.5)
    combine_type = config["model"]["combine_type"]
    lon_lat_inputs = config["model"]["lon_lat_inputs"]

    # --- Bayesian optimization search phase ---
    # Reuses SphericalBayesianOptimizer / train_model_process from DRF
    # unmodified, but — unlike exp4 — with a REAL held-out val split
    # instead of reusing train_dataset as val_data.
    optimizer = SphericalBayesianOptimizer(
        model_class=DeepMaternRandomPhaseS2RFFNN,
        train_data=train_dataset,
        val_data=val_dataset,
        val_data2=val_dataset,  # accepted by SphericalBayesianOptimizer but never referenced internally
        test_data=test_dataset_unlabeled,
        num_layers=config["model"]["num_layers"],
        spatial_input_dim=3,
        temporal_input_dim=1,
        hidden_dim=config["model"]["hidden_dim"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        output_dim=config["model"]["output_dim"],
        nu=nu,
        num_models=config["training"]["num_models"],
        d_phi=d_phi,
        d_theta=d_theta,
        device=config["device"],
        p_weight=config["training"]["p_weight"],
        n_iterations=config["bayesian_optimization"]["n_iterations"],
        n_initial_samples=config["bayesian_optimization"]["initial_samples"],
        n_epochs=config["training"]["num_epochs"],
    )

    best_hyperparams, best_loss = optimizer.optimize(
        n_iterations=config["bayesian_optimization"]["n_iterations"]
    )

    spatial_lengthscale, temporal_lengthscale, amplitude, lengthscale2, amplitude2 = best_hyperparams
    best_loss_value = best_loss.item() if torch.is_tensor(best_loss) else best_loss
    print("Best hyperparameters:")
    print(f"  spatial_lengthscale = {spatial_lengthscale.item()}")
    print(f"  temporal_lengthscale = {temporal_lengthscale.item()}")
    print(f"  amplitude = {amplitude.item()}")
    print(f"  lengthscale2 = {lengthscale2.item()}")
    print(f"  amplitude2 = {amplitude2.item()}")
    print(f"Best validation loss: {best_loss_value}")

    # --- Final retrain ---
    # No experiment in this codebase saves trained weights, so getting an
    # actual (state_dict, predictions) pair for the winning hyperparameters
    # means retraining once more with them.
    print("Retraining final ensemble with winning hyperparameters (to obtain saveable weights)...")
    num_models = config["training"]["num_models"]
    model_config = dict(
        num_layers=config["model"]["num_layers"],
        spatial_input_dim=3,
        temporal_input_dim=1,
        hidden_dim=config["model"]["hidden_dim"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        output_dim=config["model"]["output_dim"],
        nu=nu,
        lon_lat_inputs=lon_lat_inputs,
        combine_type=combine_type,
    )

    final_models = []
    for seed in range(num_models):
        model = train_final_model(
            model_class=DeepMaternRandomPhaseS2RFFNN,
            train_data=train_dataset,
            hyperparams=best_hyperparams,
            seed=seed,
            device=device,
            n_epochs=config["training"]["num_epochs"],
            batch_size=config["training"]["batch_size"],
            **model_config,
        )
        final_models.append(model)

    # --- Predict on the REAL held-out test set with the final ensemble ---
    with torch.no_grad():
        test_spatial = spatial_X_test.to(device)
        test_temporal = temporal_X_test.to(device)
        per_model_preds = torch.stack(
            [model(test_spatial, test_temporal).cpu() for model in final_models]
        )  # [num_models, N_test, 1]

    mean_pred = per_model_preds.mean(dim=0)
    var_pred = per_model_preds.var(dim=0)

    # --- Actual test-set accuracy metrics against real labels ---
    # This is the specific gap exp4 has: it loads a labeled test set and
    # never scores against it.
    rmse = compute_rmse(mean_pred, y_test)
    nlpd = compute_nlpd(mean_pred, var_pred, y_test)
    crps = compute_crps(mean_pred, var_pred, y_test)
    print(f"Test-set RMSE: {rmse:.4f}")
    print(f"Test-set NLPD: {nlpd:.4f}")
    print(f"Test-set CRPS: {crps:.4f}")

    # --- Output paths, auto-derived from the config filename ---
    # Never hand-typed in the config itself — removes the copy-paste-bug
    # class already hit twice with exp3/exp4's configs (missing
    # plot_mean_filename, mistyped tensor path).
    results_dir = SCRIPT_DIR / "results" / config_path.stem
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([{
        "spatial_lengthscale": round(spatial_lengthscale.item(), 4),
        "temporal_lengthscale": round(temporal_lengthscale.item(), 4),
        "amplitude": round(amplitude.item(), 4),
        "lengthscale2": round(lengthscale2.item(), 4),
        "amplitude2": round(amplitude2.item(), 4),
        "best_val_loss": round(best_loss_value, 4),
        "test_rmse": round(rmse, 4),
        "test_nlpd": round(nlpd, 4),
        "test_crps": round(crps, 4),
    }])
    results_df.to_csv(results_dir / "results.csv", index=False)

    torch.save(mean_pred, results_dir / "final_predictions.pt")
    torch.save(var_pred, results_dir / "final_variance.pt")
    torch.save(per_model_preds, results_dir / "individual_final_predictions.pt")

    for seed, model in enumerate(final_models):
        save_checkpoint(
            path=results_dir / "checkpoints" / f"model_{seed}.pt",
            state_dict=model.state_dict(),
            model_config=model_config,
            hyperparameters=best_hyperparams,
            norm_stats=data["normalization_stats"],
            seed=seed,
        )

    print(f"Results, predictions, and checkpoints saved to {results_dir}")

    # --- Plots ---
    # Scatter, not imshow — SWOT/altimeter tracks are sparse and irregular,
    # not a dense grid (exp4 has a leftover duplicate imshow block from
    # copy-pasting exp3's gridded-data plotting code; not repeating that
    # here). Spatial data is raw radians (never z-scored), so converting
    # back to degrees for plotting is a plain rad2deg — no normalization to
    # reverse, unlike exp4.
    test_lons = torch.rad2deg(spatial_X_test[:, 0]).numpy()
    test_lats = torch.rad2deg(spatial_X_test[:, 1]).numpy()

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)})
    mean_plot = ax.scatter(
        test_lons, test_lats,
        c=mean_pred.squeeze().numpy(),
        cmap="coolwarm",
        s=15,
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
    var_plot = ax.scatter(
        test_lons, test_lats,
        c=var_pred.squeeze().numpy(),
        cmap="viridis",
        s=15,
        transform=ccrs.PlateCarree(),
        vmin=0,
        vmax=0.2,
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    plt.colorbar(var_plot, ax=ax, orientation="horizontal", pad=0.05, label="Variance")
    plt.savefig(results_dir / "final_variance.png", dpi=300)
    plt.close(fig)

    print("Done.")
