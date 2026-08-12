"""
Save/load trained DRF model checkpoints.

No experiment in this codebase (exp1-4) ever saved trained weights before —
every prior run retrained from scratch whenever it needed a model's
predictions on new data. This module exists so exp5 can persist a trained
ensemble once and reuse it later (e.g. against future KaRIn data) without
retraining.

A checkpoint bundles everything needed to reconstruct and correctly apply
the model, not just its weights: the trained state_dict, the architecture
config needed to reconstruct the model class, the winning hyperparameters
(required at construction time), the normalization stats used to preprocess
this model's training data (new data must be normalized the same way before
being fed to the loaded model), and the seed for provenance.
"""

from pathlib import Path

import torch


def save_checkpoint(path, state_dict, model_config, hyperparameters, norm_stats, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "model_config": model_config,
            "hyperparameters": hyperparameters,
            "normalization_stats": norm_stats,
            "seed": seed,
        },
        path,
    )


def load_checkpoint(path):
    return torch.load(path, weights_only=False, map_location="cpu")