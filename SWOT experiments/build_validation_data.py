"""
Builds a validation config's core artifacts: DRF whole-globe grid snapshots
(one per drf_run x day in the config's date range) and the central
points_validation.parquet table (DRF/DUACS sampled at real SWOT points).
This is the "build" half of the DRF/DUACS/SWOT comparison pipeline -- it
only computes and saves data, it never displays anything. A separate,
later script will consume these already-computed artifacts to make the
actual comparison plots (grid diff maps, zoomed swath overlays, spatial
error binning, variance-vs-error) -- deliberately kept out of this script,
mirroring replot_grid.py's/plot_search_history.py's existing split from
experiment_5.py: this script does the expensive GPU/DUACS/SWOT work once
and saves everything needed; plotting is pure downstream consumption, no
model/DUACS/SWOT loading required there at all.

For each day in [start_date, end_date] (the validation config's own range
-- independent of which days SWOT happens to cover, since a grid
snapshot's DRF-vs-DUACS comparison doesn't need SWOT at all), and for each
configured drf_run:
  - kind: fixed -- one checkpoint set, reused for every day; only the
    predicted date changes. Loaded once, not once per day.
  - kind: sliding_window -- a different checkpoint set per day
    (results_dir_pattern with {date} substituted), one inference per day.
Either way, if that day's checkpoints don't exist yet -- not trained on
SSH yet, an expected, normal state given how incrementally this pipeline
runs, not an error -- that (drf_run, day) pair is skipped with a clear
message rather than raising.

Each grid is saved as grid_snapshots/<label>/<date>_mean.pt +
_variance.pt (raw arrays, reused for point-sampling below without
recomputing) and matching .png images (same coolwarm/viridis conventions
as experiment_5.py's/replot_grid.py's final_mean.png/final_variance.png).
Skipped if already saved, so a long multi-day run can be resumed like
everything else in this pipeline (--force to rebuild anyway).

points_validation.parquet is then built by sampling DUACS (nearest-
neighbor lookup, existing query_l4_points) and each drf_run's ALREADY-
SAVED grid (nearest-neighbor grid-index lookup) at every real SWOT point
within [start_date, end_date] -- not recomputing any inference. A drf_run
with no saved grid for a given day (skipped above) simply contributes no
rows for that day; the table is allowed to be incomplete while training is
still in progress elsewhere.

Usage:
    python build_validation_data.py --config configs/validation/val_01july25_sliding.yaml
    python build_validation_data.py --config configs/validation/val_01july25_sliding.yaml --force
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from copernicus_l4_pipeline import load_l4_dataset, query_l4_points
from drive_paths import get_drive_base_path
from replot_grid import _NUM_LATS, _NUM_LONGS, _save_global_grid_plot, build_grid_inputs, load_ensemble, predict_grid

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
RESULTS_DIR = SCRIPT_DIR / "results"
VALIDATION_RESULTS_DIR = SCRIPT_DIR / "validation_results"


def _find_config_by_name(search_dir, name):
    for path in sorted(search_dir.rglob("*.yaml")):
        with open(path) as f:
            candidate = yaml.safe_load(f)
        if candidate and candidate.get("name") == name:
            return candidate
    raise FileNotFoundError(f"No config with name={name!r} found under {search_dir}")


def _load_swot_points(swot_config_name, drive_base):
    path = drive_base / "parquet files" / "swot_l2" / swot_config_name / "points.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No SWOT points found at {path} -- run build_swot_l2_data.py first")
    return pd.read_parquet(path)


def _drf_run_label(drf_run):
    return drf_run.get("label") or f"{drf_run.get('results_dir', drf_run.get('results_dir_pattern'))}__{drf_run['candidate']}"


def _resolve_candidate_dir(results_dir, requested_candidate):
    """Returns the first of [requested_candidate, "final_loss_and_val_rmse_winner"]
    that actually has a checkpoints/ dir with real checkpoints in it --
    experiment_5.py collapses final_loss_winner/val_rmse_winner into a
    single final_loss_and_val_rmse_winner folder whenever the same round
    wins both selection criteria (see e.g.
    exp_july25_2daywindow_locked/2025-07-01), so a config requesting
    val_rmse_winner on a run that collapsed needs this fallback rather
    than a hard miss. Falls back to the originally-requested path if
    neither exists, so the caller's own existence check still reports a
    real, meaningful path in its skip message."""
    for candidate in dict.fromkeys([requested_candidate, "final_loss_and_val_rmse_winner"]):
        checkpoints_dir = results_dir / candidate / "checkpoints"
        if checkpoints_dir.exists() and any(checkpoints_dir.glob("model_*.pt")):
            return checkpoints_dir
    return results_dir / requested_candidate / "checkpoints"


def _checkpoints_dir_for_day(drf_run, date_str):
    kind = drf_run.get("kind", "fixed")
    if kind == "fixed":
        results_dir = RESULTS_DIR / drf_run["results_dir"]
    elif kind == "sliding_window":
        results_dir = RESULTS_DIR / drf_run["results_dir_pattern"].format(date=date_str)
    else:
        raise NotImplementedError(f"drf_run kind={kind!r} not supported")
    return _resolve_candidate_dir(results_dir, drf_run["candidate"])


def _nearest_grid_index(values, grid_min, grid_max, n):
    step = (grid_max - grid_min) / (n - 1)
    idx = np.round((np.asarray(values) - grid_min) / step).astype(int)
    return np.clip(idx, 0, n - 1)


def _rmse_bias_variance(errors):
    """NaN-aware -- a handful of points can legitimately have no DUACS
    value (e.g. a coastal/masked grid cell DUACS doesn't cover; verified
    live that 1 point out of 7.8M SWOT points on a single day fell in one)
    and shouldn't null out the whole aggregate the way plain np.mean/np.var
    would via NaN-propagation."""
    errors = np.asarray(errors)
    return (
        float(np.sqrt(np.nanmean(errors ** 2))),
        float(np.nanmean(errors)),
        float(np.nanvar(errors)),
    )


def build_grid_snapshots(config, target_dates, device, out_dir, force=False):
    """For each (drf_run, day), builds and saves a DRF whole-globe grid
    snapshot -- skipping pairs whose checkpoints don't exist yet, and
    (unless force) pairs whose snapshot is already saved. Returns
    {(label, date_str): (grid_mean, grid_var)} for every pair that ended
    up available (freshly computed or already on disk), for
    build_points_table() to sample from without recomputing."""
    grid_snapshots_dir = out_dir / "grid_snapshots"
    available = {}

    for drf_run in config["drf_runs"]:
        label = _drf_run_label(drf_run)
        run_dir = grid_snapshots_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)

        models = None
        temporal_mean = temporal_std = temporal_znormalised = None
        loaded_checkpoints_dir = None

        for date_str in target_dates:
            mean_path = run_dir / f"{date_str}_mean.pt"
            var_path = run_dir / f"{date_str}_variance.pt"

            if not force and mean_path.exists() and var_path.exists():
                print(f"[{label}] {date_str}: grid already saved, loading.")
                available[(label, date_str)] = (torch.load(mean_path), torch.load(var_path))
                continue

            checkpoints_dir = _checkpoints_dir_for_day(drf_run, date_str)
            if not checkpoints_dir.exists() or not any(checkpoints_dir.glob("model_*.pt")):
                print(f"[{label}] {date_str}: no checkpoints at {checkpoints_dir}, skipping (not trained yet?).")
                continue

            # Only reload the ensemble when the checkpoints directory
            # actually changes -- always, for sliding_window (a different
            # dir per day); once, for fixed (the same dir every day).
            if checkpoints_dir != loaded_checkpoints_dir:
                print(f"[{label}] Loading checkpoints from {checkpoints_dir}...")
                models, temporal_mean, temporal_std, temporal_znormalised = load_ensemble(checkpoints_dir, device)
                loaded_checkpoints_dir = checkpoints_dir

            print(f"[{label}] {date_str}: predicting grid...")
            grid_spatial_X, grid_temporal_X = build_grid_inputs(date_str, temporal_mean, temporal_std, temporal_znormalised)
            grid_mean, grid_var = predict_grid(models, device, grid_spatial_X, grid_temporal_X)

            torch.save(grid_mean, mean_path)
            torch.save(grid_var, var_path)
            _save_global_grid_plot(
                grid_mean.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
                colorbar_label=f"DRF Predicted SLA (m) -- {date_str}", save_path=run_dir / f"{date_str}_mean.png",
            )
            _save_global_grid_plot(
                grid_var.T.numpy(), cmap="viridis", vmin=0, vmax=0.2,
                colorbar_label=f"DRF Variance -- {date_str}", save_path=run_dir / f"{date_str}_variance.png",
            )
            available[(label, date_str)] = (grid_mean, grid_var)

    return available


def build_points_table(config, target_dates, swot_df, duacs_ds, grid_snapshots, out_dir):
    swot_df = swot_df[swot_df["day"].dt.strftime("%Y-%m-%d").isin(target_dates)].copy()

    print("Looking up DUACS at every SWOT point (nearest-neighbor, space+time)...")
    swot_df["duacs_val"] = query_l4_points(
        duacs_ds, swot_df["lon"].values, swot_df["lat"].values, swot_df["time"].values,
    )
    swot_df["duacs_run"] = config["duacs_config"]
    swot_df["duacs_error"] = swot_df["duacs_val"] - swot_df["swot_ssha"]

    row_groups = []
    for drf_run in config["drf_runs"]:
        label = _drf_run_label(drf_run)

        day_dfs = []
        for date_str, day_rows in swot_df.groupby(swot_df["day"].dt.strftime("%Y-%m-%d")):
            if (label, date_str) not in grid_snapshots:
                print(f"[{label}] {date_str}: no grid available, excluding {len(day_rows)} point(s) from the table.")
                continue
            grid_mean_np, grid_var_np = (t.numpy() for t in grid_snapshots[(label, date_str)])

            lon_idx = _nearest_grid_index(day_rows["lon"].values, -180, 180, _NUM_LONGS)
            lat_idx = _nearest_grid_index(day_rows["lat"].values, -90, 90, _NUM_LATS)

            day_rows = day_rows.copy()
            day_rows["drf_run"] = label
            day_rows["drf_pred"] = grid_mean_np[lon_idx, lat_idx]
            day_rows["drf_var"] = grid_var_np[lon_idx, lat_idx]
            day_dfs.append(day_rows)

        if day_dfs:
            run_df = pd.concat(day_dfs, ignore_index=True)
            run_df["drf_error"] = run_df["drf_pred"] - run_df["swot_ssha"]
            row_groups.append(run_df)

    if not row_groups:
        print("WARNING: no drf_run had any available grid for any day -- points_validation.parquet will have no DRF rows.")
        points_validation_df = swot_df.assign(drf_run=None, drf_pred=np.nan, drf_var=np.nan, drf_error=np.nan)
    else:
        points_validation_df = pd.concat(row_groups, ignore_index=True)

    points_validation_df = points_validation_df[[
        "lon", "lat", "time", "cycle", "pass", "swot_ssha",
        "duacs_run", "duacs_val", "duacs_error",
        "drf_run", "drf_pred", "drf_var", "drf_error",
    ]]
    points_validation_df.to_parquet(out_dir / "points_validation.parquet", index=False)
    print(f"Saved {len(points_validation_df)} row(s) to {out_dir / 'points_validation.parquet'}")

    return points_validation_df, swot_df


def main():
    parser = argparse.ArgumentParser(
        description="Build DRF grid snapshots and the SWOT-anchored points_validation table for a validation config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a configs/validation/*.yaml file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Rebuild grid snapshots even if already saved")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device = torch.device(args.device)

    out_dir = VALIDATION_RESULTS_DIR / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(config["start_date"], config["end_date"], freq="1D")]
    print(f"Validation config '{config['name']}': {len(target_dates)} target day(s) ({config['start_date']} to {config['end_date']}).")

    drive_base, _ = get_drive_base_path()

    print(f"Loading SWOT points for '{config['swot_config']}'...")
    swot_df = _load_swot_points(config["swot_config"], drive_base)
    swot_df["day"] = pd.to_datetime(swot_df["time"]).dt.floor("D")

    print(f"Loading DUACS config '{config['duacs_config']}'...")
    duacs_config = _find_config_by_name(CONFIGS_DIR / "data_l4", config["duacs_config"])
    duacs_ds = load_l4_dataset(duacs_config)

    print("Building DRF grid snapshots...")
    grid_snapshots = build_grid_snapshots(config, target_dates, device, out_dir, force=args.force)

    print("Building points_validation.parquet...")
    points_validation_df, swot_df_filtered = build_points_table(config, target_dates, swot_df, duacs_ds, grid_snapshots, out_dir)

    summary_rows = []
    duacs_rmse, duacs_bias, duacs_var = _rmse_bias_variance(swot_df_filtered["duacs_error"])
    summary_rows.append({
        "run": config["duacs_config"], "kind": "duacs", "n_points": len(swot_df_filtered),
        "rmse": duacs_rmse, "bias": duacs_bias, "variance_of_diff": duacs_var,
    })
    for label, run_df in points_validation_df.groupby("drf_run"):
        if pd.isna(label):
            continue
        rmse, bias, var = _rmse_bias_variance(run_df["drf_error"])
        summary_rows.append({
            "run": label, "kind": "drf", "n_points": len(run_df),
            "rmse": rmse, "bias": bias, "variance_of_diff": var,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("rmse", ascending=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Saved {out_dir / 'summary.csv'}")
    print(summary_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
