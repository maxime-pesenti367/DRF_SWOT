import torch
import yaml
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torch.utils.data import TensorDataset
from DRF.models import DeepMaternRandomPhaseS2RFFNN
from DRF.spherical_uq_methods import SphericalBayesianOptimizer, train_model_process
from pathlib import Path

if __name__ == '__main__':
    # Get the directory where experiment_4.py lives
    SCRIPT_DIR = Path(__file__).resolve().parent
    config_path = SCRIPT_DIR / "configs" / "config_exp4.yaml"

    # 1. Load Configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    device = torch.device(config["device"])

    # 2. LOAD YOUR SAVED TENSORS (Added weights_only=False to suppress warning)
    print("Loading pre-processed data...")
    data = torch.load(config["data"]["tensor_data_path"], weights_only=False)

    spatial_X_train = data['spatial_X_train']
    temporal_X_train = data['temporal_X_train']
    y_train = data['y_train'].unsqueeze(-1)  # Forces shape to [N, 1]

    spatial_X_test = data['spatial_X_test']
    temporal_X_test = data['temporal_X_test']
    y_test = data['y_test'].unsqueeze(-1)    # Forces shape to [N, 1]

    # 3. WRAP IN TENSOR DATASETS
    # Note: experiment_3 expects spatial, temporal, and target in one dataset for training
    train_dataset = TensorDataset(spatial_X_train, temporal_X_train, y_train)

    # experiment_3 uses a test dataset with only spatial and temporal inputs
    test_dataset = TensorDataset(spatial_X_test, temporal_X_test) 

    # 4. CALCULATE GRID SPACING (d_phi, d_theta)
    # Assuming spatial_X_train[:, 0] is Longitude and [:, 1] is Latitude
    lons = spatial_X_train[:, 0]
    lats = spatial_X_train[:, 1]
    unique_lons = torch.unique(lons)
    unique_lats = torch.unique(lats)
    d_phi = torch.abs(unique_lons[1] - unique_lons[0]).item() if len(unique_lons) > 1 else 0.0
    d_theta = torch.abs(unique_lats[1] - unique_lats[0]).item() if len(unique_lats) > 1 else 0.0

    # 5. INITIALIZE MODEL & OPTIMIZER
    optimizer = SphericalBayesianOptimizer(
        model_class=DeepMaternRandomPhaseS2RFFNN,
        train_data=train_dataset,
        val_data=train_dataset, 
        val_data2=train_dataset, 
        test_data=test_dataset,
        num_layers=config["model"]["num_layers"],
        spatial_input_dim=3, 
        temporal_input_dim=1,
        hidden_dim=config["model"]["hidden_dim"],
        bottleneck_dim=config["model"]["bottleneck_dim"],
        output_dim=config["model"]["output_dim"],
        nu=config["model"]["kwargs"].get("nu", 1.5),
        num_models=config["training"]["num_models"],
        d_phi=d_phi,
        d_theta=d_theta,
        device=config["device"],
        p_weight=config["training"]["p_weight"],
        n_iterations=config["bayesian_optimization"]["n_iterations"],
        n_initial_samples=config["bayesian_optimization"]["initial_samples"],
        n_epochs=config["training"]["num_epochs"],
    )

    # 6. OPTIMIZE AND SAVE 
    best_hyperparams, best_loss = optimizer.optimize(
        n_iterations=config["bayesian_optimization"]["n_iterations"]
    )

    (
        spatial_lengthscale,
        temporal_lengthscale,
        amplitude,
        lengthscale2,
        amplitude2,
    ) = best_hyperparams
    
    print(f"Best hyperparameters:")
    print(f"  spatial_lengthscale = {spatial_lengthscale.item()}")
    print(f"  temporal_lengthscale = {temporal_lengthscale.item()}")
    print(f"  amplitude = {amplitude.item()}")
    print(f"  lengthscale2 = {lengthscale2.item()}")
    print(f"  amplitude2 = {amplitude2.item()}")
    print(f"Best validation loss: {best_loss.item()}")

    top_prediction = sorted(
        optimizer.test_predictions_per_iteration, key=lambda x: x[1]
    )[:1]

    def extract_tensors_and_params(pred_tuple):
        return pred_tuple[0], pred_tuple[2]

    def process_prediction_set(tensor_tuple):
        # Convert each element to a PyTorch tensor if it's not already one
        tensor_tuple = [
            torch.tensor(t) if isinstance(t, np.ndarray) else t
            for t in tensor_tuple
        ]
        return torch.stack(tensor_tuple)

    extracted_predictions_and_params = [
        extract_tensors_and_params(pred) for pred in top_prediction
    ]
    processed_predictions = [
        process_prediction_set(pred[0]) for pred in extracted_predictions_and_params
    ]
    
    print("Shape of processed predictions:", processed_predictions[0].shape)
    final_test_predictions = processed_predictions[0].mean(dim=0)
    var_final_pred = processed_predictions[0].var(dim=0)

    import pandas as pd

    nll_hyperparams_list = []
    for pred, nll, hyperparams in top_prediction:
        entry = {"NLL": nll}
        if isinstance(hyperparams, dict):
            entry.update(hyperparams)
        elif isinstance(hyperparams, (list, tuple)):
            for i, param in enumerate(hyperparams):
                entry[f"param_{i}"] = (
                    param.item() if torch.is_tensor(param) else param
                )
        else:
            entry["param"] = hyperparams
        nll_hyperparams_list.append(entry)

    df = pd.DataFrame(nll_hyperparams_list)
    csv_filename = config["results"]["csv_filename"]
    df.to_csv(csv_filename, index=False)
    print(f"NLL and hyperparameters saved to {csv_filename}")

    torch.save(final_test_predictions, config["results"]["predictions_filename"])
    torch.save(var_final_pred, config["results"]["variance_filename"])
    torch.save(
        processed_predictions[0],
        config["results"]["individual_predictions_filename"],
    )
    
    _NUM_LONGS = 512
    _NUM_LATS = 256
    print("Final test predictions saved.")






    import matplotlib.pyplot as plt

    # --- NEW: Un-normalize coordinates for plotting ---
    # We grab the stats we safely saved in the dictionary earlier
    spatial_mean = data['normalization_stats']['spatial_mean'].cpu()
    spatial_std = data['normalization_stats']['spatial_std'].cpu()
    
    # Move tensors to CPU and flatten them for plotting
    test_coords_norm = spatial_X_test.cpu()
    test_preds = final_test_predictions.cpu().squeeze()
    test_vars = var_final_pred.cpu().squeeze()

    # 1. Reverse the Z-score normalization
    test_coords_rad = (test_coords_norm * spatial_std) + spatial_mean
    
    # 2. Convert radians back to degrees
    test_lons = torch.rad2deg(test_coords_rad[:, 0]).numpy()
    test_lats = torch.rad2deg(test_coords_rad[:, 1]).numpy()
    # --------------------------------------------------

    # Plot 1: Mean Predictions
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
    )
    # We use ax.scatter() instead of ax.imshow() for sparse track data
    ssha_plot = ax.scatter(
        test_lons, test_lats, 
        c=test_preds.numpy(), 
        cmap="coolwarm", 
        s=15, # Controls the size of the dots
        transform=ccrs.PlateCarree(),
        vmin=-0.25, 
        vmax=0.25
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    
    output_path_nn = config["results"]["plot_mean_filename"]
    plt.colorbar(ssha_plot, ax=ax, orientation="horizontal", pad=0.05, label="Predicted SLA (m)")
    plt.savefig(output_path_nn, dpi=300)
    plt.show()

    # Plot 2: Variance (Uncertainty)
    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
    )
    variance_plot = ax.scatter(
        test_lons, test_lats, 
        c=test_vars.numpy(), 
        cmap="viridis",  
        s=15,
        transform=ccrs.PlateCarree(),
        vmin=0,  
        vmax=0.2
    )
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    
    output_path_variance = config["results"]["plot_variance_filename"]
    plt.colorbar(variance_plot, ax=ax, orientation="horizontal", pad=0.05, label="Variance")
    plt.savefig(output_path_variance, dpi=300)
    plt.show()

    
    # ------------------------------------------------------------------
    # Whole-globe snapshot: forward-pass the trained ensemble over a dense
    # global lon/lat grid at a single reference time (the dataset's mean
    # timestamp -> normalized time = 0.0), instead of only at the sparse
    # along-track test points plotted above. No model weights were saved
    # during optimization, so this retrains the ensemble with the winning
    # hyperparameters and points its inference step at the grid instead.
    # ------------------------------------------------------------------
    print("Building global grid for whole-globe snapshot...")

    grid_lons_deg = torch.linspace(-180, 180, _NUM_LONGS)
    grid_lats_deg = torch.linspace(-90, 90, _NUM_LATS)
    grid_lon_grid, grid_lat_grid = torch.meshgrid(
        grid_lons_deg, grid_lats_deg, indexing="ij"
    )
    grid_coords_rad = torch.stack(
        [
            torch.deg2rad(grid_lon_grid.reshape(-1)),
            torch.deg2rad(grid_lat_grid.reshape(-1)),
        ],
        dim=1,
    )
    grid_spatial_X = ((grid_coords_rad - spatial_mean) / spatial_std).to(device)
    grid_temporal_X = torch.zeros(
        grid_spatial_X.shape[0], 1, device=device
    )  # normalized mean time
    grid_dataset = TensorDataset(grid_spatial_X, grid_temporal_X)

    grid_predictions = []
    for seed in range(config["training"]["num_models"]):
        _, _, _, grid_preds = train_model_process(
            model_class=DeepMaternRandomPhaseS2RFFNN,
            train_data=train_dataset,
            val_data=train_dataset,
            test_data=grid_dataset,
            num_layers=config["model"]["num_layers"],
            spatial_input_dim=3,
            temporal_input_dim=1,
            hidden_dim=config["model"]["hidden_dim"],
            bottleneck_dim=config["model"]["bottleneck_dim"],
            output_dim=config["model"]["output_dim"],
            spatial_rff_layer_type="MaternRandomPhaseS2RFFLayer",
            temporal_rff_layer_type="Matern",
            hyperparams=best_hyperparams,
            seed=seed,
            device=device,
            nu=config["model"]["kwargs"].get("nu", 1.5),
            d_phi=d_phi,
            d_theta=d_theta,
            n_epochs=config["training"]["num_epochs"],
        )
        grid_predictions.append(torch.tensor(grid_preds))

    grid_predictions = torch.stack(grid_predictions)  # [num_models, N_grid, 1]
    grid_mean_pred = grid_predictions.mean(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    grid_var_pred = grid_predictions.var(dim=0).squeeze().reshape(_NUM_LONGS, _NUM_LATS)
    print("Global grid predictions computed.")

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
    )
    ssha_plot = ax.imshow(
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
    plt.colorbar(ssha_plot, ax=ax, orientation="horizontal", pad=0.05, label="Predicted SLA (m)")
    plt.savefig(config["results"]["plot_mean_grid_filename"], dpi=300)
    plt.show()

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=0)}
    )
    variance_plot = ax.imshow(
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
    plt.colorbar(variance_plot, ax=ax, orientation="horizontal", pad=0.05, label="Variance")
    plt.savefig(config["results"]["plot_variance_grid_filename"], dpi=300)
    plt.show()