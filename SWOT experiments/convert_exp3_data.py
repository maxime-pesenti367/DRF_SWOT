"""
Converts exp3's flat CSV dataset (lon_20_ku, lat_20_ku, time, ssha_20_ku)
into exp5-compatible .pt tensor files, so exp5's training pipeline
(experiment_5.py) can be run against exp3's actual data for direct
comparison -- e.g. to test whether exp5's rougher-looking maps trace back to
its temporal encoding (continuous z-scored Unix seconds) rather than to the
underlying data itself.

Deliberately standalone from build_experiment_data.py, which is built around
the Copernicus-fetch pipeline (heavy xarray/copernicusmarine deps, meant to
run from the separate fetch venv -- see CLAUDE.md). This script only needs
pandas/torch/numpy, so it runs from the main venv, same as experiment_5.py.

Never touches experiment_3.py/data_utils.py -- only reads exp3's CSV.

Usage:
    python convert_exp3_data.py                        # raw temporal (matches exp3's own convention)
    python convert_exp3_data.py --normalize-temporal    # z-scored temporal (matches exp5's convention)
    python convert_exp3_data.py --experiment-name exp3_dataset_v2 --seed 7
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_splits import random_split
from drive_paths import get_drive_base_path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_DIR.parent / "DeepRandomFeatures" / "gdrive_data" / "exp3.csv"

# Matches get_spherical_data()'s split (DeepRandomFeatures/src/DRF/data_utils.py)
# -- data_splits.random_split's own docstring already claims to match
# exp3/exp4's methodology, so these are the same proportions, not a new choice.
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.7, 0.15, 0.15


def tensorize_split(df, normalize_temporal, temporal_mean, temporal_std):
    spatial = torch.tensor(
        np.deg2rad(df[["lon_20_ku", "lat_20_ku"]].to_numpy()), dtype=torch.float32
    )
    temporal_raw = torch.tensor(df[["time"]].to_numpy(), dtype=torch.float32)
    target = torch.tensor(df["ssha_20_ku"].to_numpy(), dtype=torch.float32)

    if normalize_temporal:
        temporal = (temporal_raw - temporal_mean) / (temporal_std + 1e-8)
    else:
        temporal = temporal_raw

    return spatial, temporal, target


def main():
    parser = argparse.ArgumentParser(
        description="Convert exp3's CSV dataset into exp5-compatible .pt tensor files"
    )
    parser.add_argument(
        "--csv-path", type=str, default=str(DEFAULT_CSV_PATH),
        help="Path to exp3's CSV (default: DeepRandomFeatures/gdrive_data/exp3.csv)",
    )
    parser.add_argument(
        "--experiment-name", type=str, default="exp3_dataset",
        help="Name under 'pytorch tensors/' to save the .pt file as (default: exp3_dataset)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random split seed (default: 42)")
    parser.add_argument(
        "--normalize-temporal", action="store_true",
        help="Z-score the temporal input using the train split's own mean/std "
             "(exp5's convention). Default: leave raw, unnormalized (exp3's own convention).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"exp3 CSV not found at: {csv_path}")

    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")

    train_df, val_df, test_df = random_split(
        df, train=TRAIN_FRAC, val=VAL_FRAC, test=TEST_FRAC, seed=args.seed
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    temporal_mean = torch.tensor([0.0])
    temporal_std = torch.tensor([1.0])
    if args.normalize_temporal:
        train_temporal_raw = torch.tensor(train_df[["time"]].to_numpy(), dtype=torch.float32)
        temporal_mean = train_temporal_raw.mean(dim=0)
        temporal_std = train_temporal_raw.std(dim=0)

    spatial_X_train, temporal_X_train, y_train = tensorize_split(
        train_df, args.normalize_temporal, temporal_mean, temporal_std
    )
    spatial_X_val, temporal_X_val, y_val = tensorize_split(
        val_df, args.normalize_temporal, temporal_mean, temporal_std
    )
    spatial_X_test, temporal_X_test, y_test = tensorize_split(
        test_df, args.normalize_temporal, temporal_mean, temporal_std
    )

    tensors = {
        "spatial_X_train": spatial_X_train,
        "temporal_X_train": temporal_X_train,
        "y_train": y_train,
        "spatial_X_val": spatial_X_val,
        "temporal_X_val": temporal_X_val,
        "y_val": y_val,
        "spatial_X_test": spatial_X_test,
        "temporal_X_test": temporal_X_test,
        "y_test": y_test,
        "normalization_stats": {
            # Spatial is never normalized here either (raw radians), matching
            # build_experiment_data.py's own convention.
            "temporal_mean": temporal_mean,
            "temporal_std": temporal_std,
            "temporal_normalized": args.normalize_temporal,
        },
        "split_config": {
            "name": "random",
            "method": "random",
            "train": TRAIN_FRAC,
            "val": VAL_FRAC,
            "test": TEST_FRAC,
            "seed": args.seed,
        },
    }

    drive_base, _ = get_drive_base_path()
    experiment_dir = drive_base / "pytorch tensors" / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    out_path = experiment_dir / "random.pt"
    torch.save(tensors, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
