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
    split's mean/std, UNLESS the data config sets `time_znormalised: false`
    (defaults to true, i.e. today's behaviour, if the key is absent), in
    which case the timestamp is instead kept continuous and unnormalized,
    as days since 2000-01-01 00:00 (e.g. 2025-06-01 00:00 -> 9283.0). This
    exists because temporal_lengthscale's search bounds in configs/exp5/
    were originally calibrated against this raw scale (back in exp3) --
    switching to z-scored time without updating those bounds silently made
    the temporal kernel treat every timestamp as fully correlated (a
    lengthscale of up to 10 against a z-scored ~[-3,3] input range means
    "everything is close"), i.e. temporally useless predictions. Reverting
    to this raw scale for a given dataset restores the bounds' original
    intent, with no bounds changes needed.
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
from display_tracks import display_1D_tracks

SPATIAL_COLS = ["longitude", "latitude"]
TEMPORAL_COL = "time"
TARGET_COL = "sla_filtered"

# Reference epoch for the unnormalized (time_znormalised: false) temporal
# representation -- e.g. 2025-06-01 00:00 -> 9283.0 days.
TEMPORAL_REFERENCE_EPOCH = pd.Timestamp("2000-01-01")


def normalize_and_tensorize(train_df, val_df, test_df, split_config, time_znormalised=True):
    def spatial_tensor(df):
        radians = np.deg2rad(df[SPATIAL_COLS].to_numpy())
        return torch.tensor(radians, dtype=torch.float32)

    def temporal_raw_tensor(df):
        if time_znormalised:
            seconds = pd.to_datetime(df[TEMPORAL_COL]).astype("int64").to_numpy() / 1e9
            return torch.tensor(seconds, dtype=torch.float32).unsqueeze(-1)
        days = (pd.to_datetime(df[TEMPORAL_COL]) - TEMPORAL_REFERENCE_EPOCH) / pd.Timedelta(days=1)
        return torch.tensor(days.to_numpy(), dtype=torch.float32).unsqueeze(-1)

    def target_tensor(df):
        return torch.tensor(df[TARGET_COL].to_numpy(), dtype=torch.float32)

    spatial_X_train = spatial_tensor(train_df)
    spatial_X_val = spatial_tensor(val_df)
    spatial_X_test = spatial_tensor(test_df)

    temporal_train_raw = temporal_raw_tensor(train_df)
    temporal_val_raw = temporal_raw_tensor(val_df)
    temporal_test_raw = temporal_raw_tensor(test_df)

    # Always computed (train split's own mean/std, in whichever raw unit
    # time_znormalised implies) -- even when NOT applied to produce the
    # tensors below, experiment_5.py's/replot_grid.py's default whole-globe
    # snapshot still needs a sensible "center of the training window"
    # timestamp to predict at.
    temporal_mean = temporal_train_raw.mean(dim=0)
    temporal_std = temporal_train_raw.std(dim=0)

    if time_znormalised:
        temporal_X_train = (temporal_train_raw - temporal_mean) / (temporal_std + 1e-8)
        temporal_X_val = (temporal_val_raw - temporal_mean) / (temporal_std + 1e-8)
        temporal_X_test = (temporal_test_raw - temporal_mean) / (temporal_std + 1e-8)
    else:
        temporal_X_train = temporal_train_raw
        temporal_X_val = temporal_val_raw
        temporal_X_test = temporal_test_raw

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
            # module docstring. temporal_mean/temporal_std are always the
            # train split's own stats, in whichever raw unit
            # temporal_znormalised implies (seconds since 1970 if True, days
            # since 2000-01-01 if False) -- consumers (experiment_5.py,
            # replot_grid.py) must check temporal_znormalised before
            # deciding whether to actually invert/apply them.
            "temporal_znormalised": time_znormalised,
            "temporal_mean": temporal_mean,
            "temporal_std": temporal_std,
        },
        "split_config": split_config,
    }


def build_experiment_data(config, plot=False):
    """Fetches/splits/tensorizes one dataset from an already-loaded config
    dict. Pulled out of main() so build_sliding_window_data.py can call this
    same core pipeline in a loop (one call per generated per-day config)
    without duplicating it -- both scripts run in the same (fetch) venv, so
    a direct function call is enough, no subprocess/CLI boundary needed."""
    time_znormalised = config.get("time_znormalised", True)

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
    # this has to happen before calling it. config["name"] may itself
    # contain "/" (e.g. a sliding-window day's "july25_2daywindow/
    # 2025-07-01") -- pathlib's "/" join splits on that transparently, so
    # this nests exactly as deep as the name implies with no special-casing.
    drive_base, _ = get_drive_base_path()
    experiment_dir = drive_base / "pytorch tensors" / config["name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if plot:
        ds = combine_for_drf(data_dict, "xarray")
        display_1D_tracks(ds, config["name"], save_path=experiment_dir / "tracks_overview.png")

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

        tensors = normalize_and_tensorize(train_df, val_df, test_df, split_config, time_znormalised=time_znormalised)
        store_tensors(tensors, f"{config['name']}/{split_name}")


def main():
    parser = argparse.ArgumentParser(description="Build split, normalized DRF tensor files from a data config")
    parser.add_argument("--config", type=str, required=True, help="Path to a data config YAML file")
    parser.add_argument("--plot", action="store_true", help="Save a tracks/density overview PNG alongside the tensor files on Drive")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    build_experiment_data(config, plot=args.plot)


if __name__ == "__main__":
    main()