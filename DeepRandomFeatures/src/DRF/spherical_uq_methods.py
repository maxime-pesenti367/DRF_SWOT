import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from DRF.utils import (
    functional_regularisation_S2_batched,
    compute_rmse,
    compute_nlpd,
    compute_crps,
)
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def train_model_process(
    model_class,
    train_data,
    val_data,
    test_data,
    num_layers,
    spatial_input_dim,
    temporal_input_dim,
    hidden_dim,
    bottleneck_dim,
    output_dim,
    spatial_rff_layer_type,
    temporal_rff_layer_type,
    hyperparams,
    seed,
    device,
    nu,
    d_phi,
    d_theta,
    n_epochs,
):
    """
    Trains a single model for the spherical case using Huber loss.
    Calculates reg_loss during training, but does not add it to the training loss.
    """
    torch.manual_seed(seed)
    spatial_lengthscale, temporal_lengthscale, amplitude, lengthscale2, amplitude2 = (
        hyperparams
    )

    model = model_class(
        num_layers=num_layers,
        spatial_input_dim=spatial_input_dim,
        temporal_input_dim=temporal_input_dim,
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim,
        output_dim=output_dim,
        spatial_lengthscale=spatial_lengthscale.item(),
        temporal_lengthscale=temporal_lengthscale.item(),
        nu=nu,
        amplitude=amplitude.item(),
        lengthscale2=lengthscale2.item(),
        amplitude2=amplitude2.item(),
        lon_lat_inputs=True,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.HuberLoss(delta=0.1)

    train_loader = DataLoader(train_data, batch_size=8000, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8000)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch"
        )

        for batch in train_loop:
            batch_spatial_input, batch_temporal_input, batch_values = batch
            batch_spatial_input = batch_spatial_input.to(device)
            batch_temporal_input = batch_temporal_input.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()
            outputs = model(batch_spatial_input, batch_temporal_input)
            loss = criterion(outputs, batch_values)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(train_loss=train_loss / len(train_loader))

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    # model.eval()
    # with torch.no_grad():
    # val_loader_reg = DataLoader(val_data, batch_size=8000, shuffle=False)

    model.eval()
    predictions = []
    val_values = []
    val_loss = 0
    predictions = []
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc="Validation", unit="batch")

        for batch in val_loop:
            batch_spatial_input, batch_temporal_input, batch_values = batch
            batch_spatial_input = batch_spatial_input.to(device)
            batch_temporal_input = batch_temporal_input.to(device)
            batch_values = batch_values.to(device)
            outputs = model(batch_spatial_input, batch_temporal_input)
            batch_loss = criterion(outputs, batch_values).item()
            val_loss += batch_loss
            predictions.append(outputs.cpu())
            val_values.append(batch_values.cpu())

    predictions = torch.cat(predictions, dim=0)
    val_values = torch.cat(val_values, dim=0)
    avg_val_loss = val_loss / len(val_loader)
    reg_loss = functional_regularisation_S2_batched(model, val_loader, d_phi, d_theta)

    preds = []
    grid_loader = DataLoader(test_data, batch_size=8000, shuffle=False)
    with torch.no_grad():
        for batch in tqdm(grid_loader, desc="Predicting"):
            batch_spatial_input, batch_temporal_input = batch
            batch_spatial_input = batch_spatial_input.to(device)
            batch_temporal_input = batch_temporal_input.to(device)
            batch_preds = model(batch_spatial_input, batch_temporal_input).cpu().numpy()
            preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)

    return avg_val_loss, predictions, reg_loss, preds


class SphericalBayesianOptimizer:
    """
    Bayesian optimizer for the spherical case.
    """

    def __init__(
        self,
        model_class,
        train_data,
        val_data,
        val_data2,
        test_data,
        num_layers,
        spatial_input_dim,
        temporal_input_dim,
        hidden_dim,
        bottleneck_dim,
        output_dim,
        nu,
        num_models=5,
        d_phi=None,
        d_theta=None,
        device="cpu",
        p_weight=0.95,
        n_iterations=10,
        n_initial_samples=15,
        n_epochs=1,
        max_parallel_models=None,
    ):
        self.model_class = model_class
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.num_layers = num_layers
        self.spatial_input_dim = spatial_input_dim
        self.temporal_input_dim = temporal_input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.nu = nu  # Fixed nu
        self.num_models = num_models
        self.d_phi = d_phi
        self.d_theta = d_theta
        self.device = torch.device(device)
        self.p_weight = p_weight
        self.n_iterations = n_iterations
        self.n_initial_samples = n_initial_samples
        self.n_epochs = n_epochs
        # Caps how many of the num_models ensemble members train
        # concurrently in the multiprocessing pool below. Each concurrent
        # worker holds its own CUDA context, so GPU memory scales with this
        # number, not with num_models directly. Defaults to num_models
        # (fully parallel, original behaviour) so existing callers
        # (experiment_3.py/experiment_4.py) are unaffected unless they
        # explicitly opt in to a lower value.
        self.max_parallel_models = (
            max_parallel_models if max_parallel_models is not None else num_models
        )
        self.test_predictions_per_iteration = []
        # Per-round record of every hyperparameter candidate tried during
        # optimize() (both the random initial_samples and the GP-guided
        # n_iterations), kept separate from test_predictions_per_iteration
        # above so that list's existing tuple structure -- and every
        # existing reader of it, e.g. experiment_3.py's
        # `for pred, nll, hyperparams in top_prediction:` -- stays
        # completely unchanged. Populated in objective_function(); consumed
        # by experiment_5.py (and, for free, any future experiment built on
        # this same optimizer) to save/plot search progress.
        self.search_history = []

    def objective_function(self, hyperparams):
        """
        Objective function for Bayesian optimization.
        Calculates the combined validation loss and regularization loss.
        """
        # Only set the start method once per process, not once per BO round --
        # multiprocessing's own docs say this should be called at most once;
        # calling it repeatedly with force=True re-initializes the
        # multiprocessing context on every round, a plausible contributor to
        # the "leaked semaphore objects" warnings seen after multi-round runs.
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        # Prepare arguments for multiprocessing
        args_list = [
            (
                self.model_class,
                self.train_data,
                self.val_data,
                self.test_data,
                self.num_layers,
                self.spatial_input_dim,
                self.temporal_input_dim,
                self.hidden_dim,
                self.bottleneck_dim,
                self.output_dim,
                "MaternRandomPhaseS2RFFLayer",  # spatial_rff_layer_type
                "Matern",  # temporal_rff_layer_type
                hyperparams,
                seed,
                self.device,
                self.nu,
                self.d_phi,
                self.d_theta,
                self.n_epochs,
            )
            for seed in range(self.num_models)
        ]

        # mp.Pool(processes=k) with k < len(args_list) already queues the
        # remaining tasks and runs them as earlier workers finish -- so
        # capping `processes` at max_parallel_models is enough to bound
        # peak concurrent GPU memory; no extra batching logic needed.
        # Using close()+join() instead of the Pool context manager
        # (which calls terminate() on exit) so worker processes -- and
        # their CUDA contexts -- are torn down gracefully and memory is
        # fully released before the next Bayesian-optimization round starts.
        pool = mp.Pool(processes=self.max_parallel_models)
        try:
            results = pool.starmap(train_model_process, args_list)
        finally:
            pool.close()
            pool.join()

        val_losses, all_predictions, reg_losses, test_predictions = zip(*results)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_reg_loss = sum(reg_losses) / len(reg_losses)
        avg_predictions = torch.stack(all_predictions).mean(dim=0).to(self.device)
        # Ensemble variance across the num_models members' val predictions --
        # computed here purely for the val_nlpd/val_crps tracking below, same
        # source tensors as avg_predictions just above, no extra training.
        var_predictions = torch.stack(all_predictions).var(dim=0).to(self.device)
        val_loader = DataLoader(self.val_data, batch_size=900)

        all_val_values = []
        for batch in val_loader:
            _, _, val_values_batch = batch
            all_val_values.append(val_values_batch)
        val_values = torch.cat(all_val_values, dim=0).to(self.device)
        huber_loss = F.huber_loss(
            avg_predictions, val_values, reduction="mean", delta=0.1
        )
        p_weight = self.p_weight
        final_loss = (1 - p_weight) * huber_loss.item() + p_weight * avg_reg_loss
        # final_loss = (1 - p_weight) * avg_val_loss + p_weight * avg_reg_loss

        self.test_predictions_per_iteration.append(
            (test_predictions, final_loss, hyperparams)
        )

        def to_float(x):
            return x.item() if torch.is_tensor(x) else x

        spatial_lengthscale, temporal_lengthscale, amplitude, lengthscale2, amplitude2 = hyperparams
        self.search_history.append({
            "round": len(self.search_history),
            "spatial_lengthscale": to_float(spatial_lengthscale),
            "temporal_lengthscale": to_float(temporal_lengthscale),
            "amplitude": to_float(amplitude),
            "lengthscale2": to_float(lengthscale2),
            "amplitude2": to_float(amplitude2),
            "huber_loss": huber_loss.item(),
            "avg_reg_loss": avg_reg_loss,
            "final_loss": final_loss,
            "val_rmse": compute_rmse(avg_predictions, val_values),
            "val_nlpd": compute_nlpd(avg_predictions, var_predictions, val_values),
            "val_crps": compute_crps(avg_predictions, var_predictions, val_values),
        })

        return final_loss

    def optimize(self, n_iterations=10):
        """
        Runs the Bayesian optimization process.
        """

        bounds = torch.tensor(
            [[1e-5, 1e-5, 1e-5, 1e-5, 1e-5], [0.1, 10, 1, 10, 1]],
            dtype=torch.float64,
            device=self.device,
        )

        num_initial_points = self.n_initial_samples
        initial_samples = torch.rand(num_initial_points, 5, device=self.device)
        train_x = bounds[0] + (bounds[1] - bounds[0]) * initial_samples
        train_y = torch.tensor(
            [self.objective_function(x) for x in train_x],
            dtype=torch.float64,
            device=self.device,
        ).unsqueeze(-1)
        n_iterations = self.n_iterations

        for i in range(n_iterations):
            # input_transform/outcome_transform put the GP surrogate's own
            # internal fit on a well-scaled footing: the 5 hyperparameters
            # span wildly different raw ranges (e.g. spatial_lengthscale
            # 1e-5-0.1 vs temporal_lengthscale 1e-5-10), which otherwise lets
            # the wider dimensions dominate the GP's distance metric and
            # starves it of sensitivity to the narrower ones; final_loss
            # values can also span many orders of magnitude across
            # candidates. Both transforms are applied/reversed automatically
            # by botorch wherever the model is used below (posterior calls,
            # best_f comparisons, optimize_acqf's candidate search) -- every
            # other line in this loop keeps operating in raw, original units
            # unchanged.
            model = SingleTaskGP(
                train_x,
                train_y,
                input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            EI = ExpectedImprovement(model, best_f=train_y.min().item())
            candidate, _ = optimize_acqf(
                EI,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
            )

            new_y = torch.tensor(
                [self.objective_function(candidate.squeeze())],
                dtype=torch.float64,
                device=self.device,
            ).unsqueeze(-1)
            train_x = torch.cat([train_x, candidate])
            train_y = torch.cat([train_y, new_y])

            print(f"Iteration {i+1}: Best loss = {train_y.min().item()}")

        best_idx = train_y.argmin()
        return train_x[best_idx], train_y[best_idx]
