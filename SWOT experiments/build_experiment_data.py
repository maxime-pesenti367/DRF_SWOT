"""
Builds one pre-split, pre-normalized .pt tensor file per split strategy
listed in a data config.

Usage:
    python build_experiment_data.py --config configs/data/all_sats_1_day.yaml

Orchestrates the *existing* fetch/load/combine pipeline from
copernicus_pipeline.py, then applies one of data_splits.py's split
strategies, then normalizes and tensorizes the result itself (does NOT
reuse process_and_split_dataframe — that function bakes together a 2-way
split with spatial z-score normalization, which is exactly the geometry bug
this pipeline exists to avoid; see exp5_implementation_plan.md).

Normalization applied here:
  - Spatial: degrees -> radians only. No z-scoring. spherical_to_cartesian()
    in layers.py needs true angles; z-scoring breaks that geometry.
  - Temporal: datetime -> Unix epoch seconds -> z-scored using the TRAIN
    split's mean/std.
  - Target (sla_filtered): left raw, unnormalized (matches existing
    convention in copernicus_pipeline.process_and_split_dataframe).
"""

import argparse

import numpy as np
import pandas as pd
import torch
import yaml

from copernicus_pipeline import (
    fetch_and_store_satellites,
    load_experiment_dataset,
    combine_for_drf,
    store_tensors,
    get_drive_base_path,
)
from data_splits import SPLIT_METHODS

SPATIAL_COLS = ["longitude", "latitude"]
TEMPORAL_COL = "time"
TARGET_COL = "sla_filtered"


def normalize_and_tensorize(train_df, val_df, test_df, split_config):
    def spatial_tensor(df):
        radians = np.deg2rad(df[SPATIAL_COLS].to_numpy())
        return torch.tensor(radians, dtype=torch.float32)

    def temporal_raw_tensor(df):
        seconds = pd.to_datetime(df[TEMPORAL_COL]).astype("int64").to_numpy() / 1e9
        return torch.tensor(seconds, dtype=torch.float32).unsqueeze(-1)

    def target_tensor(df):
        return torch.tensor(df[TARGET_COL].to_numpy(), dtype=torch.float32)

    spatial_X_train = spatial_tensor(train_df)
    spatial_X_val = spatial_tensor(val_df)
    spatial_X_test = spatial_tensor(test_df)

    temporal_train_raw = temporal_raw_tensor(train_df)
    temporal_val_raw = temporal_raw_tensor(val_df)
    temporal_test_raw = temporal_raw_tensor(test_df)

    temporal_mean = temporal_train_raw.mean(dim=0)
    temporal_std = temporal_train_raw.std(dim=0)

    temporal_X_train = (temporal_train_raw - temporal_mean) / (temporal_std + 1e-8)
    temporal_X_val = (temporal_val_raw - temporal_mean) / (temporal_std + 1e-8)
    temporal_X_test = (temporal_test_raw - temporal_mean) / (temporal_std + 1e-8)

    return {
        "spatial_X_train": spatial_X_train,
        "temporal_X_train": temporal_X_train,
        "y_train": target_tensor(train_df),
        "spatial_X_val": spatial_X_val,
        "temporal_X_val": temporal_X_val,
        "y_val": target_tensor(val_df),
        "spatial_X_test": spatial_X_test,
        "temporal_X_test": temporal_X_test,
        "y_test": target_tensor(test_df),
        "normalization_stats": {
            # Spatial is intentionally NOT normalized (raw radians) — see
            # module docstring. Only temporal has real stats to record.
            "temporal_mean": temporal_mean,
            "temporal_std": temporal_std,
        },
        "split_config": split_config,
    }


def main():
    parser = argparse.ArgumentParser(description="Build split, normalized DRF tensor files from a data config")
    parser.add_argument("--config", type=str, required=True, help="Path to a data config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Fetching/loading data for '{config['name']}' ({config['start_date']} to {config['end_date']})...")
    fetch_and_store_satellites(config)
    data_dict = load_experiment_dataset(config)
    df = combine_for_drf(data_dict, "pandas")
    print(f"Combined dataframe has {len(df)} rows.")

    # One subfolder per experiment under "pytorch tensors/", matching the
    # results/<experiment>/ convention used everywhere else in this project
    # (results/exp1/, results/exp3/, experiment_5.py's results/<config>/,
    # etc.) instead of a flat pile of double-underscore-named files.
    # store_tensors() doesn't create missing parent directories itself, so
    # this has to happen before calling it.
    drive_base, _ = get_drive_base_path()
    (drive_base / "pytorch tensors" / config["name"]).mkdir(parents=True, exist_ok=True)

    for split_config in config["splits"]:
        split_name = split_config["name"]
        method = split_config["method"]
        if method not in SPLIT_METHODS:
            raise ValueError(f"Unknown split method '{method}' for split '{split_name}'")

        split_kwargs = {k: v for k, v in split_config.items() if k not in ("name", "method")}
        train_df, val_df, test_df = SPLIT_METHODS[method](df, **split_kwargs)

        print(
            f"[{split_name}] split sizes -> "
            f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
        )
        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print(
                f"[{split_name}] WARNING: at least one split is empty. "
                f"This is expected/acceptable for the 1-day smoke-test scope "
                f"(see exp5_implementation_plan.md §6) but the resulting "
                f".pt file will not be usable for training/eval as-is."
            )

        tensors = normalize_and_tensorize(train_df, val_df, test_df, split_config)
        store_tensors(tensors, f"{config['name']}/{split_name}")


if __name__ == "__main__":
    main()