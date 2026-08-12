# Pure train/val/test split strategies for a combined satellite DataFrame
# (the output of copernicus_pipeline.combine_for_drf(..., "pandas")).
#
# No I/O, no config parsing here on purpose — build_experiment_data.py owns
# reading the config and dispatching to the right function by `method`.

import numpy as np
import pandas as pd


def random_split(df, train, val, test, seed):
    """Plain random point-wise split (matches exp3/exp4's current behaviour,
    and the DRF paper's own methodology). Not block-aware — see
    temporal_block_split / spatial_block_split for splits that avoid
    spatial/temporal autocorrelation leakage between train and test."""
    assert abs(train + val + test - 1.0) < 1e-6, "train/val/test must sum to 1.0"

    rng = np.random.default_rng(seed)
    n = len(df)
    indices = rng.permutation(n)

    train_end = int(train * n)
    val_end = train_end + int(val * n)

    train_df = df.iloc[indices[:train_end]].reset_index(drop=True)
    val_df = df.iloc[indices[train_end:val_end]].reset_index(drop=True)
    test_df = df.iloc[indices[val_end:]].reset_index(drop=True)
    return train_df, val_df, test_df


def temporal_block_split(df, test_start_date, val_fraction, seed, time_col="time"):
    """Everything at/after test_start_date goes to test (a genuine
    extrapolation-to-a-later-time holdout). val is a random val_fraction
    sample of the remaining (pre-cutoff) data."""
    time_values = pd.to_datetime(df[time_col])
    test_cutoff = pd.Timestamp(test_start_date)

    test_mask = time_values >= test_cutoff
    test_df = df[test_mask].reset_index(drop=True)
    remaining_df = df[~test_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_remaining = len(remaining_df)
    indices = rng.permutation(n_remaining)
    val_end = int(val_fraction * n_remaining)

    val_df = remaining_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = remaining_df.iloc[indices[val_end:]].reset_index(drop=True)
    return train_df, val_df, test_df


def spatial_block_split(df, test_bbox, val_fraction, seed, lon_col="longitude", lat_col="latitude"):
    """All points inside test_bbox (a lon/lat box) go to test (a genuine
    extrapolation-to-an-unseen-region holdout). val is a random val_fraction
    sample of the remaining (outside-box) data."""
    test_mask = (
        (df[lon_col] >= test_bbox["lon_min"]) & (df[lon_col] <= test_bbox["lon_max"]) &
        (df[lat_col] >= test_bbox["lat_min"]) & (df[lat_col] <= test_bbox["lat_max"])
    )
    test_df = df[test_mask].reset_index(drop=True)
    remaining_df = df[~test_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_remaining = len(remaining_df)
    indices = rng.permutation(n_remaining)
    val_end = int(val_fraction * n_remaining)

    val_df = remaining_df.iloc[indices[:val_end]].reset_index(drop=True)
    train_df = remaining_df.iloc[indices[val_end:]].reset_index(drop=True)
    return train_df, val_df, test_df


SPLIT_METHODS = {
    "random": random_split,
    "temporal_block": temporal_block_split,
    "spatial_block": spatial_block_split,
}