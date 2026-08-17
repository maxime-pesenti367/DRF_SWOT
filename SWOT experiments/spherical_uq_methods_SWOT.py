"""
SWOT-project variant of DRF.spherical_uq_methods, adding optional early
stopping / LR scheduling / gradient clipping to the Bayesian-optimization
search phase.

A deliberate duplication of train_model_process, not a modification of it --
experiment_3.py/experiment_4.py depend on that function (and
SphericalBayesianOptimizer) staying exactly as-is for a stable baseline (see
CLAUDE.md). SphericalBayesianOptimizerSWOT subclasses the original rather
than copying it wholesale: __init__ and optimize() are inherited unchanged
(optimize() only ever calls self.objective_function, which resolves
polymorphically to the override below), so only objective_function needs its
own copy -- unavoidable since it has to route the search phase through
train_model_process_swot and pass the extra settings into each worker
process.

experiment_5.py only uses this module when a config sets one of
training.early_stopping / training.lr_scheduler / training.gradient_clipping;
otherwise it uses DRF.spherical_uq_methods.SphericalBayesianOptimizer
unmodified, exactly as before.
"""

import copy

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from DRF.spherical_uq_methods import SphericalBayesianOptimizer
from DRF.utils import (
    functional_regularisation_S2_batched,
    compute_rmse,
    compute_nlpd,
    compute_crps,
)


def train_model_process_swot(
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
    early_stopping_patience=None,
    lr_scheduler_config=None,
    gradient_clip_max_norm=None,
):
    """
    Trains a single model for the spherical case using Huber loss -- same as
    DRF.spherical_uq_methods.train_model_process, plus three optional,
    independently-opt-in features:

      - early_stopping_patience: stop once this many epochs pass with no
        improvement in per-epoch val Huber loss, then roll back to the best
        epoch's weights before the (expensive, still one-shot) reg_loss /
        predictions below are computed -- so a candidate that spiked into
        instability near the end of training doesn't get scored on that
        spike. Requires a per-epoch val pass, which the original function
        doesn't do (it only evaluates once, after all epochs finish); that
        val pass is cheap (forward-only) and separate from reg_loss's
        autograd-based computation, which stays one-shot regardless.
      - lr_scheduler_config: {"type": "cosine"} or
        {"type": "plateau", "patience": N}.
      - gradient_clip_max_norm: caps the L2 norm of all gradients combined
        before each optimizer step.

    All three default to None, reproducing the original function's behaviour
    exactly when a config doesn't opt into any of them.
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

    scheduler = None
    scheduler_type = None
    if lr_scheduler_config is not None:
        scheduler_type = lr_scheduler_config.get("type", "cosine")
        if scheduler_type == "cosine":
            # T_max tied to this run's own n_epochs, not a separately-guessed
            # value -- the decay always spans exactly the intended budget
            # regardless of how many epochs a given config uses (1 to 30
            # across existing configs).
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
            if gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(train_loss=train_loss / len(train_loader))

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Cheap per-epoch val signal for early stopping / plateau scheduling
        # only -- the expensive autograd-based reg_loss stays a one-shot
        # computation after the loop, same as the original function.
        if early_stopping_patience is not None or scheduler_type == "plateau":
            model.eval()
            epoch_val_loss_sum = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch_spatial_input, batch_temporal_input, batch_values = batch
                    batch_spatial_input = batch_spatial_input.to(device)
                    batch_temporal_input = batch_temporal_input.to(device)
                    batch_values = batch_values.to(device)
                    outputs = model(batch_spatial_input, batch_temporal_input)
                    epoch_val_loss_sum += criterion(outputs, batch_values).item()
            epoch_val_loss = epoch_val_loss_sum / len(val_loader)

            if scheduler_type == "plateau":
                scheduler.step(epoch_val_loss)

            if early_stopping_patience is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
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

        if scheduler_type == "cosine":
            scheduler.step()

    if early_stopping_patience is not None and best_state is not None:
        model.load_state_dict(best_state)

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


class SphericalBayesianOptimizerSWOT(SphericalBayesianOptimizer):
    """
    Adds optional early_stopping_patience / lr_scheduler_config /
    gradient_clip_max_norm to SphericalBayesianOptimizer's search phase, by
    routing through train_model_process_swot instead of train_model_process.
    See this module's docstring for why this subclasses rather than modifies
    the shared DRF class.
    """

    def __init__(
        self,
        *args,
        early_stopping_patience=None,
        lr_scheduler_config=None,
        gradient_clip_max_norm=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_config = lr_scheduler_config
        self.gradient_clip_max_norm = gradient_clip_max_norm

    def objective_function(self, hyperparams):
        """
        Same as SphericalBayesianOptimizer.objective_function, except calling
        train_model_process_swot (with the extra settings) instead of
        train_model_process, and printing which BO round is about to start
        (self.search_history's current length is exactly the count of
        already-completed rounds, i.e. the index of this one -- same logic
        experiment_5.py uses to derive round_type from position alone).
        See objective_function's DRF counterpart for the rationale behind
        everything else here -- unchanged.
        """
        round_idx = len(self.search_history)
        if round_idx < self.n_initial_samples:
            print(f"--- Initial sample {round_idx + 1}/{self.n_initial_samples} ---")
        else:
            iteration_num = round_idx - self.n_initial_samples + 1
            print(f"--- BO iteration {iteration_num}/{self.n_iterations} ---")

        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
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
                self.early_stopping_patience,
                self.lr_scheduler_config,
                self.gradient_clip_max_norm,
            )
            for seed in range(self.num_models)
        ]

        pool = mp.Pool(processes=self.max_parallel_models)
        try:
            results = pool.starmap(train_model_process_swot, args_list)
        finally:
            pool.close()
            pool.join()

        val_losses, all_predictions, reg_losses, test_predictions = zip(*results)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_reg_loss = sum(reg_losses) / len(reg_losses)
        avg_predictions = torch.stack(all_predictions).mean(dim=0).to(self.device)
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

        # Same bias / variance-of-difference decomposition of RMSE added to
        # experiment_5.py's final test-set scoring (val_rmse**2 ~=
        # val_bias**2 + val_variance_of_diff) -- here on the val set instead,
        # once per BO round. SWOT-only (not added to the protected DRF
        # SphericalBayesianOptimizer/objective_function this subclasses),
        # so search_history.csv only has these two columns for runs that
        # went through this class -- plot_search_history.py checks for their
        # presence rather than assuming they're always there.
        val_residuals = avg_predictions - val_values
        val_bias = val_residuals.mean().item()
        val_variance_of_diff = val_residuals.var().item()

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
            "val_bias": val_bias,
            "val_variance_of_diff": val_variance_of_diff,
        })

        return final_loss
