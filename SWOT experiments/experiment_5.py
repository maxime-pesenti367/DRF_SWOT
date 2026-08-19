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
import copy
import math
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
from spherical_uq_methods_SWOT import SphericalBayesianOptimizerSWOT
from DRF.utils import compute_rmse, compute_nlpd, compute_crps
from drive_paths import get_drive_base_path
from model_io import save_checkpoint
from plot_search_history import plot_search_progress, plot_training_curve

SCRIPT_DIR = Path(__file__).resolve().parent

# Whole-globe grid snapshot resolution + figure sizing, shared by the
# final_mean.png/final_variance.png plots below. Matplotlib's default figure
# sizing (a fixed inches x dpi canvas, map axes auto-shrunk by
# fig.colorbar(ax=...) to share space with the colorbar) silently
# downsamples/blends the underlying grid to whatever pixel budget the
# canvas happens to have -- so bumping _NUM_LONGS/_NUM_LATS alone doesn't
# actually buy more real detail in the saved PNG. _save_global_grid_plot
# below instead gives the map its own explicitly-sized axes so every grid
# cell maps to exactly one output pixel, no resampling.
_NUM_LONGS = 2880  # 0.125 deg/pixel: 360 / 0.125
_NUM_LATS = 1440   # 0.125 deg/pixel: 180 / 0.125
# 256 so _NUM_LONGS/_GRID_DPI, _NUM_LATS/_GRID_DPI, and the colorbar strip /
# border added below all reduce to exact binary fractions (power-of-2
# denominator, e.g. 1440/256 = 5.625 = 45/8) -- avoids inches<->pixel
# rounding nudging the map axes a fraction of a pixel off its target
# footprint, same as when this was 2048x1024.
_GRID_DPI = 256
_COLORBAR_HEIGHT_PX = 256  # legend space only, not pixel-critical
_BORDER_PX = 32  # small uniform white margin around the whole saved image

_TOTAL_WIDTH_PX = _NUM_LONGS + 2 * _BORDER_PX
_TOTAL_HEIGHT_PX = _NUM_LATS + _COLORBAR_HEIGHT_PX + 2 * _BORDER_PX
_FIG_WIDTH_IN = _TOTAL_WIDTH_PX / _GRID_DPI
_FIG_HEIGHT_IN = _TOTAL_HEIGHT_PX / _GRID_DPI


def _save_global_grid_plot(data_np, cmap, vmin, vmax, colorbar_label, save_path):
    """Saves a whole-globe grid snapshot with the map rendered at exactly
    _NUM_LONGS x _NUM_LATS pixels -- no interpolation/resampling -- inset by
    a uniform _BORDER_PX white margin on all four sides of the saved image.
    The map gets its own explicitly-sized axes, positioned by exact pixel
    offsets into the full canvas, instead of matplotlib's default
    auto-layout, which shrinks the axes to make room for the colorbar and
    resamples the data to whatever pixel footprint is left over."""
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
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    # Colorbar sits in the border-to-map gap below -- not pixel-critical,
    # so centered with generous margins rather than exactly positioned.
    cax_left = map_left + 0.15 * map_width
    cax_width = 0.7 * map_width
    cax_bottom = _BORDER_PX / _TOTAL_HEIGHT_PX + 0.35 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
    cax_height = 0.3 * (_COLORBAR_HEIGHT_PX / _TOTAL_HEIGHT_PX)
    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    fig.colorbar(im, cax=cax, orientation="horizontal", label=colorbar_label)
    fig.savefig(save_path, dpi=_GRID_DPI)
    plt.close(fig)


def _round_sig(x, sig=6):
    """Rounds to `sig` significant figures rather than decimal places.
    Appropriate for the CSVs this feeds -- they mix wildly different
    magnitudes in the same table (e.g. spatial_lengthscale down around 1e-5
    alongside val_nlpd in the hundreds), where a fixed round(x, ndigits)
    would zero out the small values while being needlessly verbose for the
    large ones."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def _round_df_sig_figs(df, sig=6):
    """Applies _round_sig to every numeric column of df; non-numeric columns
    (e.g. round_type, is_winner) pass through untouched."""
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].apply(lambda x: _round_sig(x, sig))
    return df


def train_final_model(
    model_class,
    train_data,
    val_data,
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
    early_stopping_patience=None,
    lr_scheduler_config=None,
    gradient_clip_max_norm=None,
):
    """
    Trains one model to completion and returns the model object itself (not
    just its predictions), so its weights can be saved afterwards -- plus a
    per-epoch train/val loss history, since this is the actual model whose
    weights get checkpointed and used going forward, and previously had zero
    logging at all (unlike the throwaway BO-search-phase models, which at
    least print per-epoch loss to the terminal).

    Deliberately NOT routed through DRF.spherical_uq_methods.train_model_process
    — that function is reused as-is for the Bayesian-optimization search
    phase below, but it only ever returns predictions, never the trained
    model, and extending its return contract would require also touching
    experiment_3.py/experiment_4.py's existing calls to it, which is out of
    scope here. This is a small, deliberate duplication of its training loop
    for the two new things it doesn't support: handing back a model whose
    weights can actually be persisted, and a loss history for it.

    early_stopping_patience/lr_scheduler_config/gradient_clip_max_norm are
    optional opt-in features (see spherical_uq_methods_SWOT.py for the
    BO-search-phase equivalent) -- all default to None, reproducing exactly
    this function's original behaviour when a config doesn't set them.
    Deliberately does NOT try to reproduce whatever epoch the BO-search
    phase's early stopping picked for these same hyperparameters -- GPU
    training isn't bit-reproducible by default, so this applies the same
    stopping *rule* fresh to its own run rather than replaying a specific
    epoch number from a different run.
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
    val_loader = DataLoader(val_data, batch_size=batch_size)

    scheduler = None
    scheduler_type = None
    if lr_scheduler_config is not None:
        scheduler_type = lr_scheduler_config.get("type", "cosine")
        if scheduler_type == "cosine":
            # T_max tied to this run's own n_epochs -- see
            # spherical_uq_methods_SWOT.py's copy of this for the rationale.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=lr_scheduler_config.get("patience", 2)
            )
        else:
            raise ValueError(f"Unknown lr_scheduler type: {scheduler_type!r}")

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    epoch_history = []
    for epoch in range(n_epochs):
        model.train()
        train_loss_sum = 0.0
        n_train_batches = 0
        for batch in train_loader:
            batch_spatial, batch_temporal, batch_values = batch
            batch_spatial = batch_spatial.to(device)
            batch_temporal = batch_temporal.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()
            outputs = model(batch_spatial, batch_temporal)
            loss = criterion(outputs, batch_values)
            loss.backward()
            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            optimizer.step()

            train_loss_sum += loss.item()
            n_train_batches += 1
        avg_train_loss = train_loss_sum / n_train_batches

        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch_spatial, batch_temporal, batch_values = batch
                batch_spatial = batch_spatial.to(device)
                batch_temporal = batch_temporal.to(device)
                batch_values = batch_values.to(device)
                outputs = model(batch_spatial, batch_temporal)
                val_loss_sum += criterion(outputs, batch_values).item()
                n_val_batches += 1
                val_preds.append(outputs.cpu())
                val_targets.append(batch_values.cpu())
        avg_val_loss = val_loss_sum / n_val_batches
        val_rmse = compute_rmse(torch.cat(val_preds), torch.cat(val_targets))

        epoch_history.append({
            "seed": seed,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_rmse": val_rmse,
        })

        if scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        elif scheduler_type == "cosine":
            scheduler.step()

        if early_stopping_patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stopping_patience} epochs)"
                )
                break

    if early_stopping_patience is not None and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, epoch_history


def _retrain_and_save_candidate(
    hyperparams,
    criterion_label,
    results_dir,
    config,
    model_config,
    train_dataset,
    val_dataset,
    spatial_X_test,
    temporal_X_test,
    y_test,
    normalization_stats,
    device,
    early_stopping_patience,
    lr_scheduler_config,
    gradient_clip_max_norm,
):
    """Retrains config['training']['num_models'] fresh models on
    `hyperparams`, scores them against the real held-out test set, and saves
    everything (checkpoints, predictions, training curve, whole-globe grid
    plots) into results_dir/<criterion_label>/. Returns a dict of that
    candidate's metrics for the shared results.csv row.

    Pulled out of main() so it can run once per selected BO candidate --
    previously there was always exactly one (the final_loss winner); now
    main() calls this for each of the final_loss winner and val_rmse winner
    (or just once if they're the same round -- see main()'s same_winner
    check)."""
    candidate_dir = results_dir / criterion_label
    candidate_dir.mkdir(parents=True, exist_ok=True)

    num_models = config["training"]["num_models"]
    final_models = []
    training_curve_records = []
    for seed in range(num_models):
        model, epoch_history = train_final_model(
            model_class=DeepMaternRandomPhaseS2RFFNN,
            train_data=train_dataset,
            val_data=val_dataset,
            hyperparams=hyperparams,
            seed=seed,
            device=device,
            n_epochs=config["training"]["num_epochs"],
            batch_size=config["training"]["batch_size"],
            early_stopping_patience=early_stopping_patience,
            lr_scheduler_config=lr_scheduler_config,
            gradient_clip_max_norm=gradient_clip_max_norm,
            **model_config,
        )
        final_models.append(model)
        training_curve_records.extend(epoch_history)

    # --- Predict on the REAL held-out test set with this candidate's ensemble ---
    # Batched (not one giant forward pass) -- large test sets (e.g. exp3's
    # ~1.2M-row test split) blow up GPU memory otherwise: the spherical
    # layer's RandomPhaseFeatureMap materializes a (hidden_dim, N) tensor
    # internally, which for hidden_dim=1000 and N in the millions is
    # multiple GiB in one allocation.
    test_loader = DataLoader(
        TensorDataset(spatial_X_test, temporal_X_test),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    with torch.no_grad():
        per_model_preds_list = []
        for model in final_models:
            batch_preds = []
            for batch_spatial, batch_temporal in test_loader:
                batch_preds.append(
                    model(batch_spatial.to(device), batch_temporal.to(device)).cpu()
                )
            per_model_preds_list.append(torch.cat(batch_preds, dim=0))
        per_model_preds = torch.stack(per_model_preds_list)  # [num_models, N_test, 1]

    mean_pred = per_model_preds.mean(dim=0)
    var_pred = per_model_preds.var(dim=0)

    rmse = compute_rmse(mean_pred, y_test)
    nlpd = compute_nlpd(mean_pred, var_pred, y_test)
    crps = compute_crps(mean_pred, var_pred, y_test)
    residuals = mean_pred - y_test
    test_bias = residuals.mean().item()
    test_variance_of_diff = residuals.var().item()
    print(
        f"[{criterion_label}] Test-set RMSE: {rmse:.4f}  NLPD: {nlpd:.4f}  "
        f"CRPS: {crps:.4f}  bias: {test_bias:.4f}  variance_of_diff: {test_variance_of_diff:.4f}"
    )

    training_curve_df = _round_df_sig_figs(pd.DataFrame(training_curve_records))
    training_curve_df.to_csv(candidate_dir / "training_curve.csv", index=False)

    torch.save(mean_pred, candidate_dir / "final_predictions.pt")
    torch.save(var_pred, candidate_dir / "final_variance.pt")
    torch.save(per_model_preds, candidate_dir / "individual_final_predictions.pt")

    for seed, model in enumerate(final_models):
        save_checkpoint(
            path=candidate_dir / "checkpoints" / f"model_{seed}.pt",
            state_dict=model.state_dict(),
            model_config=model_config,
            hyperparameters=hyperparams,
            norm_stats=normalization_stats,
            seed=seed,
        )

    # --- Whole-globe grid snapshot ---
    # Dense global grid + imshow, matching exp3/exp4's whole-globe snapshot
    # style rather than scattering over the sparse real test-set points --
    # judged more informative for visualizing overall model behaviour, with
    # real accuracy (RMSE/NLPD/CRPS above) already handled separately against
    # the genuine held-out test set. No retrain needed here: final_models are
    # already in memory (and checkpointed to disk above), so we just
    # forward-pass those over the grid. Spatial inputs are raw radians here
    # (never z-scored), so no normalization needs to be applied/reversed for
    # the grid coordinates either.
    print(f"[{criterion_label}] Building global grid for whole-globe snapshot...")
    grid_lons_deg = torch.linspace(-180, 180, _NUM_LONGS)
    grid_lats_deg = torch.linspace(-90, 90, _NUM_LATS)
    grid_lon_grid, grid_lat_grid = torch.meshgrid(grid_lons_deg, grid_lats_deg, indexing="ij")
    grid_spatial_X = torch.stack(
        [torch.deg2rad(grid_lon_grid.reshape(-1)), torch.deg2rad(grid_lat_grid.reshape(-1))],
        dim=1,
    ).to(device)
    # Normalized time = 0.0 -> the training set's mean timestamp (temporal
    # data is z-scored using train-split mean/std; see normalization_stats).
    grid_temporal_X = torch.zeros(grid_spatial_X.shape[0], 1, device=device)

    grid_loader = DataLoader(
        TensorDataset(grid_spatial_X.cpu(), grid_temporal_X.cpu()),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    with torch.no_grad():
        grid_per_model_preds_list = []
        for model in final_models:
            batch_preds = []
            for batch_spatial, batch_temporal in grid_loader:
                batch_preds.append(
                    model(batch_spatial.to(device), batch_temporal.to(device)).cpu()
                )
            grid_per_model_preds_list.append(torch.cat(batch_preds, dim=0))
        grid_per_model_preds = torch.stack(grid_per_model_preds_list)  # [num_models, N_grid, 1]

    grid_mean_pred = grid_per_model_preds.mean(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    grid_var_pred = grid_per_model_preds.var(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    print(f"[{criterion_label}] Global grid predictions computed.")

    _save_global_grid_plot(
        grid_mean_pred.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
        colorbar_label="Predicted SLA (m)", save_path=candidate_dir / "final_mean.png",
    )
    _save_global_grid_plot(
        grid_var_pred.T.numpy(), cmap="viridis", vmin=0, vmax=0.2,
        colorbar_label="Variance", save_path=candidate_dir / "final_variance.png",
    )

    plot_training_curve(candidate_dir)

    def to_float(x):
        return x.item() if torch.is_tensor(x) else x

    spatial_lengthscale, temporal_lengthscale, amplitude, lengthscale2, amplitude2 = hyperparams
    return {
        "selection_criterion": criterion_label,
        "spatial_lengthscale": to_float(spatial_lengthscale),
        "temporal_lengthscale": to_float(temporal_lengthscale),
        "amplitude": to_float(amplitude),
        "lengthscale2": to_float(lengthscale2),
        "amplitude2": to_float(amplitude2),
        "test_rmse": rmse,
        "test_nlpd": nlpd,
        "test_crps": crps,
        "test_bias": test_bias,
        "test_variance_of_diff": test_variance_of_diff,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a spherical DRF model (exp5)")
    parser.add_argument("--config", type=str, required=True, help="Path to a model/training config YAML file")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["device"])

    # Seeds only the Bayesian-optimization SEARCH itself (the random initial
    # hyperparameter guesses in SphericalBayesianOptimizer.optimize(), plus
    # botorch's internal acquisition-function random restarts) so that
    # re-running the same config always explores the same candidates and
    # lands on the same winning hyperparameters. Deliberately does NOT
    # control per-ensemble-member randomness -- train_model_process() and
    # train_final_model() already call torch.manual_seed(seed) themselves
    # for each seed in range(num_models), and that diversity across seeds is
    # what gives the deep ensemble its uncertainty estimate; this call must
    # not be moved to a point where it would make those seeds redundant.
    torch.manual_seed(config.get("seed", 42))

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
    #
    # Early stopping / LR scheduling / gradient clipping are opt-in via
    # optional training.early_stopping / training.lr_scheduler /
    # training.gradient_clipping config keys -- absent by default, so an
    # existing config with none of these set behaves exactly as before,
    # using SphericalBayesianOptimizer unmodified. Setting any of them
    # switches to SphericalBayesianOptimizerSWOT (spherical_uq_methods_SWOT.py)
    # instead, which is not a fork of the DRF class but a subclass of it --
    # see that module's docstring.
    early_stopping_config = config["training"].get("early_stopping")
    early_stopping_patience = (
        early_stopping_config["patience"] if early_stopping_config else None
    )
    lr_scheduler_config = config["training"].get("lr_scheduler")
    gradient_clipping_config = config["training"].get("gradient_clipping")
    gradient_clip_max_norm = (
        gradient_clipping_config["max_norm"] if gradient_clipping_config else None
    )
    use_swot_training_features = (
        early_stopping_patience is not None
        or lr_scheduler_config is not None
        or gradient_clip_max_norm is not None
    )

    optimizer_kwargs = dict(
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
        max_parallel_models=config["training"].get("max_parallel_models"),
    )
    if use_swot_training_features:
        optimizer = SphericalBayesianOptimizerSWOT(
            early_stopping_patience=early_stopping_patience,
            lr_scheduler_config=lr_scheduler_config,
            gradient_clip_max_norm=gradient_clip_max_norm,
            # SphericalBayesianOptimizer.optimize() (the protected DRF
            # method) hardcodes its own search bounds and never reads this
            # config key at all -- SphericalBayesianOptimizerSWOT overrides
            # optimize() specifically to respect it. Only wired through on
            # this path for that reason; the protected path below is
            # unaffected either way since every existing config's bounds
            # already match the hardcoded default.
            bounds=config["bayesian_optimization"]["bounds"],
            **optimizer_kwargs,
        )
    else:
        optimizer = SphericalBayesianOptimizer(**optimizer_kwargs)
        configured_bounds = config["bayesian_optimization"]["bounds"]
        hardcoded_bounds = [[1e-5, 0.1], [1e-5, 10], [1e-5, 1], [1e-5, 10], [1e-5, 1]]
        if configured_bounds != hardcoded_bounds:
            print(
                "WARNING: this run's bayesian_optimization.bounds "
                f"{configured_bounds} will be IGNORED -- "
                "SphericalBayesianOptimizer.optimize() (the protected DRF "
                f"method) always searches the fixed range {hardcoded_bounds}. "
                "Set training.early_stopping/lr_scheduler/gradient_clipping "
                "to route through SphericalBayesianOptimizerSWOT instead, "
                "which does respect configured bounds."
            )

    optimizer.optimize(n_iterations=config["bayesian_optimization"]["n_iterations"])

    # --- Select candidates: lowest final_loss (the actual BO objective) and
    # lowest val_rmse (pure point-accuracy, ignoring the regularization term
    # final_loss also weighs in at p_weight=0.95) -- these don't always
    # agree. Both get retrained and saved; if the same round happens to win
    # both (common with small search budgets), only retrain it once.
    search_history_list = optimizer.search_history
    final_loss_winner_idx = min(
        range(len(search_history_list)), key=lambda i: search_history_list[i]["final_loss"]
    )
    val_rmse_winner_idx = min(
        range(len(search_history_list)), key=lambda i: search_history_list[i]["val_rmse"]
    )
    same_winner = final_loss_winner_idx == val_rmse_winner_idx

    def _hyperparams_from_round(round_idx):
        row = search_history_list[round_idx]
        return (
            row["spatial_lengthscale"],
            row["temporal_lengthscale"],
            row["amplitude"],
            row["lengthscale2"],
            row["amplitude2"],
        )

    if same_winner:
        candidates = [("final_loss_and_val_rmse_winner", final_loss_winner_idx)]
        print(f"Round {final_loss_winner_idx} won both final_loss and val_rmse -- one candidate to retrain.")
    else:
        candidates = [
            ("final_loss_winner", final_loss_winner_idx),
            ("val_rmse_winner", val_rmse_winner_idx),
        ]
        print(
            f"final_loss winner: round {final_loss_winner_idx}  |  "
            f"val_rmse winner: round {val_rmse_winner_idx}"
        )

    # --- Output paths, auto-derived from the config filename ---
    # Never hand-typed in the config itself — removes the copy-paste-bug
    # class already hit twice with exp3/exp4's configs (missing
    # plot_mean_filename, mistyped tensor path).
    results_dir = SCRIPT_DIR / "results" / config_path.stem
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Final retrain(s) ---
    # No experiment in this codebase saves trained weights, so getting an
    # actual (state_dict, predictions) pair for a winning hyperparameter set
    # means retraining once more with it -- once per candidate selected above.
    results_rows = []
    for criterion_label, round_idx in candidates:
        hyperparams = _hyperparams_from_round(round_idx)
        row = search_history_list[round_idx]
        print(f"--- Retraining final ensemble for {criterion_label} (round {round_idx}) ---")
        print(f"  spatial_lengthscale = {hyperparams[0]}")
        print(f"  temporal_lengthscale = {hyperparams[1]}")
        print(f"  amplitude = {hyperparams[2]}")
        print(f"  lengthscale2 = {hyperparams[3]}")
        print(f"  amplitude2 = {hyperparams[4]}")
        print(f"  final_loss (this round) = {row['final_loss']}")
        print(f"  val_rmse (this round) = {row['val_rmse']}")

        metrics = _retrain_and_save_candidate(
            hyperparams=hyperparams,
            criterion_label=criterion_label,
            results_dir=results_dir,
            config=config,
            model_config=model_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            spatial_X_test=spatial_X_test,
            temporal_X_test=temporal_X_test,
            y_test=y_test,
            normalization_stats=data["normalization_stats"],
            device=device,
            early_stopping_patience=early_stopping_patience,
            lr_scheduler_config=lr_scheduler_config,
            gradient_clip_max_norm=gradient_clip_max_norm,
        )
        metrics["best_val_loss"] = row["final_loss"]
        metrics["search_val_rmse"] = row["val_rmse"]
        results_rows.append(metrics)

    results_df = _round_df_sig_figs(pd.DataFrame(results_rows))
    results_df.to_csv(results_dir / "results.csv", index=False)

    # --- Search history: every BO candidate tried, not just the winner(s) ---
    # optimizer.search_history is populated inside objective_function() --
    # one record per hyperparameter candidate, in the order optimize() tried
    # them (first n_initial_samples random draws, then n_iterations
    # GP-guided ones), which is what lets round_type be derived purely from
    # position rather than needing any new plumbing through objective_function.
    search_history_df = pd.DataFrame(search_history_list)
    search_history_df["round_type"] = [
        "initial_sample" if i < optimizer.n_initial_samples else "iteration"
        for i in range(len(search_history_df))
    ]
    # Index-based (not a plain `== .min()` value comparison) so a tie
    # between two different rounds can't mark more than one row as the
    # winner -- matches final_loss_winner_idx/val_rmse_winner_idx above
    # exactly, which already resolve ties to a single round via Python's
    # min().
    search_history_df["is_winner"] = search_history_df.index == final_loss_winner_idx
    search_history_df["is_val_rmse_winner"] = search_history_df.index == val_rmse_winner_idx
    # Computed above from the full-precision values, so rounding for display
    # afterward doesn't affect which round either flag picks.
    search_history_df = _round_df_sig_figs(search_history_df)
    search_history_df.to_csv(results_dir / "search_history.csv", index=False)

    print(f"Results and per-candidate outputs saved to {results_dir}")

    # --- Search-progress plot ---
    # Lives only in results_dir, not duplicated into each candidate
    # subfolder -- it covers the whole search (both winners marked), so
    # there's exactly one authoritative copy to regenerate later via
    # `python plot_search_history.py --results-dir results/<config-name>`
    # (pointed at the top-level dir, not a candidate subfolder) rather than
    # redoing it once per candidate.
    plot_search_progress(results_dir)

    print("Done.")
