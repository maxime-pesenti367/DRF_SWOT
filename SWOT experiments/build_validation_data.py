"""
Builds a validation config's core artifact: the central
points_validation.parquet table (DRF/DUACS sampled at real SWOT points).
This is the "build" half of the DRF/DUACS/SWOT comparison pipeline -- it
only computes and saves data, it never displays anything. A separate,
later script will consume this table to make the actual comparison plots
(grid diff maps, zoomed swath overlays, spatial error binning,
variance-vs-error) -- deliberately kept out of this script, mirroring
replot_grid.py's/plot_search_history.py's existing split from
experiment_5.py.

This script does NOT do any DRF inference itself -- it only READS
whole-globe grid snapshots that experiment_5.py/replot_grid.py already
computed and saved in results/ (on whichever machine has the GPU):
  - kind: sliding_window -- each day trained its own model and saved its
    grid (at exactly that day's center date, via grid_snapshot_date) right
    next to its checkpoints, as final_mean_grid.pt/final_variance_grid.pt.
  - kind: fixed -- one model; its grids for many days live in a dedicated
    grids/<date>_mean.pt / <date>_variance.pt subfolder, produced by
    `replot_grid.py --start-date/--end-date`.
A (drf_run, day) with no grid file yet (not computed/synced yet -- an
expected, normal state given how incrementally this pipeline runs, not an
error) is skipped with a clear message rather than raising. This is what
makes the script safe to run somewhere with Drive access but no GPU (e.g.
a local machine, as opposed to an SSH/GPU box with poor Drive access).

Every grid used gets copied (DRF) or freshly extracted (DUACS -- it has no
pre-existing per-day grid file anywhere, just one big Zarr array) into
validation_results/<name>/grid_snapshots/<run-or-duacs-config>/<date>/,
and that folder is then the ONLY thing points_validation.parquet is built
from -- DUACS is sampled from its saved grid the same way DRF is (nearest
0.125deg pixel), not via a separate live lookup into the Zarr dataset. This
keeps a validation run fully self-contained and auditable: every value in
points_validation.parquet traces back to a specific saved file in
grid_snapshots/, never a lazily-read external dataset. DUACS's snapshot is
mean.pt/mean.png only -- it's a single deterministic product, not an
ensemble, so unlike DRF there's no variance to save. Both sides skip a
(run, day) already saved, so re-running only fills in what's new.

Grid snapshots are only ever built for target_dates -- exactly the
validation config's own start_date/end_date, nothing padded or implicitly
widened; if you want a neighbouring day available to match against, add
it to the config's date range yourself. Each source (DUACS, and
separately each drf_run) is then matched against whichever of THOSE
target_dates it actually has a saved snapshot for, independently -- not a
single day shared across sources. Different sources can still end up with
different available days if one is simply missing a snapshot for a given
target date (e.g. a drf_run that hasn't been trained/synced for that day
yet), in which case that source falls back to its own next-nearest
target_date. For every SWOT point, DUACS independently picks whichever of
ITS available days is closest, and each drf_run independently does the
same among ITS OWN available days -- so a given row's duacs_matched_date
and drf_matched_date can legitimately differ.
Both are recorded as columns in points_validation.parquet. A source with
zero saved snapshots anywhere contributes no rows at all (DUACS: a hard
error, since the table is meaningless without it; a drf_run: that run is
skipped with a clear message, other drf_runs still get built).

Every row also gets the static regional masks from masks/ (see
build_coastal_mask.py/build_variance_mask.py/build_gulf_stream_mask.py)
sampled at its own lon/lat: dist_to_coast_km/coastal_flag (coastal_mask.nc),
sla_variance_cm2/high_var_flag (sla_variance_mask.nc), and gulf_stream_flag
(gulf_stream_mask.nc). Unlike DUACS/DRF these are time-invariant, so it's a
plain nearest-pixel lookup, no per-day matching -- see _load_masks.

Usage:
    python build_validation_data.py --config configs/validation/val_01july25_sliding.yaml
"""

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede any matplotlib.pyplot import (here, transitively via
                        # replot_grid.py) -- avoids a real tkinter/Tcl crash when saving many
                        # plots in a loop (e.g. one DUACS/DRF grid PNG per day), same fix as
                        # build_sliding_window_data.py's
import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml

from copernicus_l4_pipeline import load_l4_dataset
from drive_paths import get_drive_base_path
from replot_grid import (
    _GRID_LAT_MAX, _GRID_LAT_MIN, _GRID_LON_MAX, _GRID_LON_MIN, _NUM_LATS, _NUM_LONGS, _save_global_grid_plot,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR / "configs"
RESULTS_DIR = SCRIPT_DIR / "results"
VALIDATION_RESULTS_DIR = SCRIPT_DIR / "validation_results"
MASKS_DIR = SCRIPT_DIR / "masks"


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
    that actually exists -- experiment_5.py collapses final_loss_winner/
    val_rmse_winner into a single final_loss_and_val_rmse_winner folder
    whenever the same round wins both selection criteria, so a config
    requesting val_rmse_winner on a run that collapsed needs this fallback
    rather than a hard miss. Falls back to the originally-requested path if
    neither exists, so the caller's own existence check still reports a
    real, meaningful path in its skip message."""
    for candidate in dict.fromkeys([requested_candidate, "final_loss_and_val_rmse_winner"]):
        candidate_dir = results_dir / candidate
        if candidate_dir.exists():
            return candidate_dir
    return results_dir / requested_candidate


def _grid_paths_for_day(drf_run, date_str):
    """Returns (mean_pt, var_pt, mean_png, var_png) -- reading already-
    computed grids from results/, never doing inference. See module
    docstring for the kind: fixed vs kind: sliding_window path layouts."""
    kind = drf_run.get("kind", "fixed")
    if kind == "fixed":
        candidate_dir = _resolve_candidate_dir(RESULTS_DIR / drf_run["results_dir"], drf_run["candidate"])
        grids_dir = candidate_dir / "grids"
        return (
            grids_dir / f"{date_str}_mean.pt", grids_dir / f"{date_str}_variance.pt",
            grids_dir / f"{date_str}_mean.png", grids_dir / f"{date_str}_variance.png",
        )
    if kind == "sliding_window":
        results_dir = RESULTS_DIR / drf_run["results_dir_pattern"].format(date=date_str)
        candidate_dir = _resolve_candidate_dir(results_dir, drf_run["candidate"])
        return (
            candidate_dir / "final_mean_grid.pt", candidate_dir / "final_variance_grid.pt",
            candidate_dir / "final_mean.png", candidate_dir / "final_variance.png",
        )
    raise NotImplementedError(f"drf_run kind={kind!r} not supported")


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


def build_grid_snapshots(config, target_dates, out_dir):
    """For each (drf_run, day), reads the already-computed grid .pt
    tensors from results/ -- never does inference itself (that's
    experiment_5.py's/replot_grid.py's job, on whichever machine has the
    GPU) -- and copies them into
    validation_results/<name>/grid_snapshots/<label>/<date>/ so everything
    a validation run used lives together in one self-contained place, not
    scattered back into results/. The mean.png/variance.png saved
    alongside are NOT copied from results/ -- they're freshly regenerated
    here from the same .pt tensors with mask_land=True (results/'s own
    PNGs may or may not have land masked depending on how/when they were
    produced upstream; regenerating here guarantees every DRF grid_snapshot
    PNG is land-masked consistently, without needing to touch
    experiment_5.py/replot_grid.py). Skips regenerating entirely if
    already done (checked via the destination mean.pt), so re-running only
    fills in what's new. Missing source .pt grids (not computed/synced
    yet) are skipped with a clear message. Returns {(label, date_str):
    (grid_mean, grid_var)} for every pair that's actually available, for
    build_points_table() to sample from."""
    available = {}
    for drf_run in config["drf_runs"]:
        label = _drf_run_label(drf_run)
        dest_root = out_dir / "grid_snapshots" / label

        for date_str in target_dates:
            mean_pt, var_pt, _, _ = _grid_paths_for_day(drf_run, date_str)
            if not mean_pt.exists() or not var_pt.exists():
                print(f"[{label}] {date_str}: no grid at {mean_pt}, skipping (not computed/synced yet?).")
                continue

            grid_mean, grid_var = torch.load(mean_pt), torch.load(var_pt)
            dest_day_dir = dest_root / date_str
            dest_mean_pt = dest_day_dir / "mean.pt"
            if dest_mean_pt.exists():
                print(f"[{label}] {date_str}: already copied to {dest_day_dir}, skipping.")
            else:
                dest_day_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mean_pt, dest_mean_pt)
                shutil.copy2(var_pt, dest_day_dir / "variance.pt")
                _save_global_grid_plot(
                    grid_mean.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
                    colorbar_label=f"DRF Predicted SLA (m) -- {date_str}",
                    save_path=dest_day_dir / "mean.png", mask_land=True,
                )
                _save_global_grid_plot(
                    grid_var.T.numpy(), cmap="viridis", vmin=0, vmax=0.1,
                    colorbar_label=f"DRF Variance -- {date_str}",
                    save_path=dest_day_dir / "variance.png", mask_land=True,
                )
                print(f"[{label}] {date_str}: copied grid into {dest_day_dir}")

            available[(label, date_str)] = (grid_mean, grid_var)
    return available


def save_duacs_grid_snapshots(config, target_dates, duacs_ds, out_dir):
    """Extracts DUACS's own whole-globe sla slice at each target day and
    saves it as validation_results/<name>/grid_snapshots/<duacs_config>/
    <date>/mean.pt + mean.png -- no variance.pt/variance.png, unlike the
    DRF side: DUACS is a single deterministic analysis product, not an
    ensemble, so there's no per-pixel variance to save. Skips a day
    already saved (checked via mean.pt), same resume behaviour as the DRF
    side. Transposed to (longitude, latitude) to match the orientation
    DRF's own saved grids use, so the two are directly comparable without
    remembering which one is transposed. Returns {date_str: grid_mean} for
    every target date -- loaded off disk when a date was already saved on
    a prior run, not just the freshly-extracted ones -- so
    build_points_table can sample DUACS from these in-memory tensors
    instead of touching duacs_ds again."""
    duacs_name = config["duacs_config"]
    dest_root = out_dir / "grid_snapshots" / duacs_name

    snapshots = {}
    for date_str in target_dates:
        dest_day_dir = dest_root / date_str
        mean_pt_path = dest_day_dir / "mean.pt"
        if mean_pt_path.exists():
            print(f"[{duacs_name}] {date_str}: already saved, skipping.")
            snapshots[date_str] = torch.load(mean_pt_path)
            continue

        slice_da = duacs_ds["sla"].sel(time=pd.Timestamp(date_str), method="nearest")
        grid_mean = torch.tensor(slice_da.transpose("longitude", "latitude").values, dtype=torch.float32)

        dest_day_dir.mkdir(parents=True, exist_ok=True)
        torch.save(grid_mean, mean_pt_path)
        _save_global_grid_plot(
            grid_mean.T.numpy(), cmap="coolwarm", vmin=-0.25, vmax=0.25,
            colorbar_label=f"DUACS SLA (m) -- {date_str}", save_path=dest_day_dir / "mean.png",
        )
        print(f"[{duacs_name}] {date_str}: saved grid to {dest_day_dir}")
        snapshots[date_str] = grid_mean

    return snapshots


def _closest_available_date(times, candidate_date_strs):
    """For every timestamp in `times`, returns whichever of
    candidate_date_strs is chronologically closest. Every grid snapshot
    represents a single instant at that date's midnight (see
    build_grid_inputs's target_timestamp in replot_grid.py), so "closest"
    is just minimum absolute distance to midnight of each candidate day.
    Called once per source (DUACS, each drf_run separately) against that
    source's own available dates -- see build_points_table.

    Uses searchsorted (O(N log M) time, O(N) memory) rather than a full
    (N, M) broadcast difference -- the latter allocates num_points x
    num_candidate_dates timedelta64 values, which is fine for a few-day
    validation window but allocates tens of GB and crashes
    (numpy.core._exceptions._ArrayMemoryError, hit live: 55.6 GiB for
    240,941,358 x 31) once a run covers a full month of global SWOT
    points -- candidate_date_strs is at most a few dozen values, so
    searchsorted's binary search per point is both correct and far
    cheaper than materializing the whole cross product."""
    times = pd.to_datetime(pd.Series(times)).to_numpy()
    sort_order = np.argsort(candidate_date_strs)
    candidates_sorted = pd.to_datetime(pd.Series(candidate_date_strs)).to_numpy()[sort_order]
    candidate_strs_sorted = np.array(candidate_date_strs)[sort_order]

    insert_pos = np.searchsorted(candidates_sorted, times)
    insert_pos = np.clip(insert_pos, 1, len(candidates_sorted) - 1)
    left, right = candidates_sorted[insert_pos - 1], candidates_sorted[insert_pos]
    nearest_idx = np.where((times - left) <= (right - times), insert_pos - 1, insert_pos)
    return candidate_strs_sorted[nearest_idx]


def _available_dates_for(label, grid_snapshots):
    return sorted(date_str for (lbl, date_str) in grid_snapshots if lbl == label)


def _load_masks():
    """Loads the static coastal/SLA-variance/Gulf-Stream region masks
    (see build_coastal_mask.py/build_variance_mask.py/
    build_gulf_stream_mask.py) -- all three land on the exact same
    0.125deg pixel-center grid as everything else in this pipeline
    (verified numerically against _GRID_LON_MIN/_GRID_LAT_MIN, zero
    deviation), so this is a plain nearest-pixel lookup, same as DRF/
    DUACS, not a regrid. Stored as (latitude, longitude) in their own
    files -- transposed to (longitude, latitude) here to match every grid
    tensor convention used elsewhere in this pipeline."""
    coastal_ds = xr.open_dataset(MASKS_DIR / "coastal_mask.nc")
    variance_ds = xr.open_dataset(MASKS_DIR / "sla_variance_mask.nc")
    gulf_stream_ds = xr.open_dataset(MASKS_DIR / "gulf_stream_mask.nc")
    return {
        "dist_to_coast_km": coastal_ds["distance_km"].values.T,
        "coastal_flag": coastal_ds["coastal"].values.T,
        "sla_variance_cm2": variance_ds["sla_variance"].values.T,
        "high_var_flag": variance_ds["high_var"].values.T,
        "gulf_stream_flag": gulf_stream_ds["gulf_stream"].values.T,
    }


def build_points_table(config, swot_df, grid_snapshots, out_dir):
    """Which points are IN SCOPE at all is governed by the validation
    config's own start_date/end_date. Within that scope, DUACS and each
    drf_run independently pick their own nearest available day -- see
    module docstring."""
    start_ts = pd.Timestamp(config["start_date"])
    end_ts = pd.Timestamp(config["end_date"]) + pd.Timedelta(days=1)
    swot_df = swot_df[(swot_df["time"] >= start_ts) & (swot_df["time"] < end_ts)].copy()

    # Downcast points.parquet's own columns from build_swot_l2_data.py's
    # default dtypes (lon/lat/swot_ssha float64, cycle/pass int64 -- 8
    # bytes each regardless of actual value range) to the smallest type
    # that genuinely fits, purely for this validation table's own working
    # copy (the raw points.parquet on disk is untouched). float32 easily
    # covers lon/lat/swot_ssha's real precision (already rounded to
    # 6/6/4dp at the source, and this pipeline only ever uses lon/lat for
    # 0.125deg nearest-pixel lookups -- far coarser than float32's ~7
    # significant digits). cycle/pass are always small integers (SWOT
    # never approaches int16's +-32767 range). quality_flag is dropped
    # entirely rather than downcast -- it's never read again after this
    # point and isn't part of points_validation.parquet's own schema
    # (every surviving row already has quality_flag==0 under qual_mode:
    # strict, the only mode this project currently uses, so it carries no
    # information here anyway). Measured live on the real 241M-row
    # swot_july25 table: this alone cuts the base columns from ~57 to
    # ~25 bytes/row (13.7GB -> ~6GB).
    swot_df = swot_df.drop(columns=["quality_flag"])
    swot_df["lon"] = swot_df["lon"].astype(np.float32)
    swot_df["lat"] = swot_df["lat"].astype(np.float32)
    swot_df["swot_ssha"] = swot_df["swot_ssha"].astype(np.float32)
    swot_df["cycle"] = swot_df["cycle"].astype(np.int16)
    swot_df["pass"] = swot_df["pass"].astype(np.int16)

    # lon/lat pixel index is identical for every source (masks, DUACS,
    # every drf_run) -- computed once here and reused, instead of the
    # previous version's repeated per-(source, day)-group recomputation.
    # More importantly, this also lets every new column below be built as
    # one flat array via precomputed group positions and assigned
    # directly, rather than splitting the table into ~30 per-day groups,
    # .copy()-ing each one, then pd.concat-ing them back together -- on a
    # 200M+ row table that pattern holds multiple full-size duplicates of
    # the table in memory at once and crashed live on a mere 125 MiB
    # allocation (cumulative exhaustion, not one single spike).
    print("Sampling static coastal/SLA-variance/Gulf-Stream masks at every SWOT point...")
    grid_lon_idx = _nearest_grid_index(swot_df["lon"].values, _GRID_LON_MIN, _GRID_LON_MAX, _NUM_LONGS)
    grid_lat_idx = _nearest_grid_index(swot_df["lat"].values, _GRID_LAT_MIN, _GRID_LAT_MAX, _NUM_LATS)
    masks = _load_masks()
    for col_name, mask_array in masks.items():
        swot_df[col_name] = mask_array[grid_lon_idx, grid_lat_idx]

    duacs_name = config["duacs_config"]
    duacs_available_dates = _available_dates_for(duacs_name, grid_snapshots)
    if not duacs_available_dates:
        raise RuntimeError(f"No DUACS grid snapshots available at all -- check grid_snapshots/{duacs_name}/ in {out_dir}")

    print(f"Sampling DUACS at every SWOT point (nearest of {len(duacs_available_dates)} available snapshot day(s))...")
    duacs_matched_date = _closest_available_date(swot_df["time"].values, duacs_available_dates)
    duacs_val = np.full(len(swot_df), np.nan, dtype=np.float32)
    # Iterate the small known set of candidate dates directly and use a
    # plain boolean comparison to find matching positions, rather than
    # pandas groupby/.indices -- that internally factorizes the whole
    # 200M+-element string column through a hash table to DISCOVER its
    # unique values, which is unnecessary work (and crashed live,
    # numpy.core._exceptions._ArrayMemoryError inside
    # StringHashTable.factorize) when the unique values are already known
    # ahead of time from duacs_available_dates/drf_available_dates.
    for date_str in duacs_available_dates:
        group_pos = np.flatnonzero(duacs_matched_date == date_str)
        if group_pos.size == 0:
            continue
        grid_mean_np, _ = grid_snapshots[(duacs_name, date_str)]
        grid_mean_np = grid_mean_np.numpy()
        duacs_val[group_pos] = grid_mean_np[grid_lon_idx[group_pos], grid_lat_idx[group_pos]]
    # category dtype: duacs_matched_date has <=31 distinct values and
    # duacs_run is a single repeated string across all 200M+ rows --
    # plain object dtype stores an 8-byte pointer per row regardless, so
    # this alone saves ~1.7GB/column at real scale (verified: 8 bytes/row
    # -> ~1 byte/row, since a code that small fits in a single byte).
    # Assigned in the same order as points_validation.parquet's own final
    # column order (see the end of this function) specifically so no
    # reindex/reorder step -- itself a full extra copy -- is needed later.
    # Built via from_codes, NOT pd.Categorical([duacs_name] * len(...)) --
    # that builds a 200M+-element Python list, then pandas converts it to
    # an object ndarray and runs maybe_convert_objects type-inference over
    # it, which allocates its own scratch array on top of the list and
    # object array it's inferring from (crashed live at this exact pattern
    # a few lines below, on drf_run, once more memory was resident by that
    # point in the function -- numpy.core._exceptions._ArrayMemoryError,
    # 1.80 GiB). from_codes skips all of that: a single repeated category
    # is just an all-zero int8 codes array.
    swot_df["duacs_run"] = pd.Categorical.from_codes(np.zeros(len(swot_df), dtype=np.int8), categories=[duacs_name])
    swot_df["duacs_matched_date"] = pd.Categorical(duacs_matched_date)
    swot_df["duacs_val"] = duacs_val
    # Explicit float32 -- duacs_val is float32 but swot_ssha is float64,
    # so the plain subtraction silently upcasts the result to float64
    # (doubling this column's footprint for no precision benefit -- SSHA
    # values are already only meaningful to ~4dp).
    swot_df["duacs_error"] = (swot_df["duacs_val"] - swot_df["swot_ssha"]).astype(np.float32)

    # Computed first, without building any per-run DataFrame yet -- how
    # many runs actually have data decides below whether we can avoid
    # copying swot_df at all (see below). A drf_run's own predicted
    # values are float32 already; drf_matched_date is deferred to
    # pd.Categorical only once assigned, same reasoning as duacs_matched_date.
    computed_runs = []
    for drf_run in config["drf_runs"]:
        label = _drf_run_label(drf_run)
        drf_available_dates = _available_dates_for(label, grid_snapshots)
        if not drf_available_dates:
            print(f"[{label}]: no grid snapshots available at all, excluding this run from the table.")
            continue

        # Bounds are the grid's pixel-CENTER extent (-179.9375/179.9375
        # etc, not -180/180) -- matching how the grid itself is now
        # built (see replot_grid.py's _GRID_LON_MIN comment for why
        # this fixed a real mismatch against DUACS's actual grid).
        drf_matched_date = _closest_available_date(swot_df["time"].values, drf_available_dates)
        drf_pred = np.full(len(swot_df), np.nan, dtype=np.float32)
        drf_var = np.full(len(swot_df), np.nan, dtype=np.float32)
        for date_str in drf_available_dates:
            group_pos = np.flatnonzero(drf_matched_date == date_str)
            if group_pos.size == 0:
                continue
            grid_mean_np, grid_var_np = (t.numpy() for t in grid_snapshots[(label, date_str)])
            drf_pred[group_pos] = grid_mean_np[grid_lon_idx[group_pos], grid_lat_idx[group_pos]]
            drf_var[group_pos] = grid_var_np[grid_lon_idx[group_pos], grid_lat_idx[group_pos]]

        computed_runs.append((label, drf_matched_date, drf_pred, drf_var))

    # A single drf_run (the common case) needs zero copies of swot_df --
    # its columns go straight onto swot_df in place, which then IS
    # points_validation_df. This is what actually fixed the crash this
    # was hit chasing: even after every dtype/grouping fix above, a
    # SINGLE swot_df.copy() here still failed outright (3.59 GiB, but the
    # process was already right at its ceiling from the base table's own
    # size) -- multiple drf_runs still need one copy each, since their rows
    # must genuinely coexist as separate rows in a long-format table, but
    # that's unavoidable duplication, not wasted duplication.
    #
    # copy(deep=False), NOT a plain copy() -- with >=2 drf_runs (e.g. a
    # sliding_window + a fixed run together), a deep copy() crashed too:
    # every same-dtype column gets forcibly re-consolidated into one 2D
    # block via np.vstack as part of a deep copy (pandas does this even
    # though the manager was already effectively consolidated), and that
    # vstack allocates a brand new array on top of the columns' existing
    # memory -- Unable to allocate 6.28 GiB for an array with shape
    # (7, 240941358), one float32 row per float32 column in swot_df at
    # that point (lon/lat/swot_ssha/dist_to_coast_km/sla_variance_cm2/
    # duacs_val/duacs_error). A shallow copy skips that consolidation
    # entirely (see BlockManager.copy in pandas' own source: the vstack
    # only runs `if deep`) and is safe here specifically because every
    # column set below (drf_run etc) is a NEW column name, never an
    # overwrite of an existing one -- adding a new column only appends a
    # block to the copy's own block list, it never mutates a block shared
    # with swot_df or with any other run's copy.
    row_groups = []
    for label, drf_matched_date, drf_pred, drf_var in computed_runs:
        run_df = swot_df if len(computed_runs) == 1 else swot_df.copy(deep=False)
        run_df["drf_run"] = pd.Categorical.from_codes(np.zeros(len(run_df), dtype=np.int8), categories=[label])
        run_df["drf_matched_date"] = pd.Categorical(drf_matched_date)
        run_df["drf_pred"] = drf_pred
        run_df["drf_var"] = drf_var
        run_df["drf_error"] = (run_df["drf_pred"] - run_df["swot_ssha"]).astype(np.float32)
        row_groups.append(run_df)

    if not row_groups:
        print("WARNING: no drf_run had any available grid -- points_validation.parquet will have no DRF rows.")
        points_validation_df = swot_df.assign(
            drf_run=None, drf_matched_date=None, drf_pred=np.nan, drf_var=np.nan, drf_error=np.nan
        )
    else:
        points_validation_df = pd.concat(row_groups, ignore_index=True)

    # Every column above was deliberately assigned in exactly this order,
    # so points_validation_df already has it -- no df[[...]] reindex here
    # (itself a full extra copy of the whole table, on top of everything
    # else in this function that's already been rewritten once to avoid
    # exactly that class of crash). The assert is just a cheap tripwire
    # against future edits silently drifting the assignment order above.
    expected_columns = [
        "lon", "lat", "time", "cycle", "pass", "swot_ssha", "is_leftmost",
        "dist_to_coast_km", "coastal_flag", "sla_variance_cm2", "high_var_flag", "gulf_stream_flag",
        "duacs_run", "duacs_matched_date", "duacs_val", "duacs_error",
        "drf_run", "drf_matched_date", "drf_pred", "drf_var", "drf_error",
    ]
    assert list(points_validation_df.columns) == expected_columns, (
        f"points_validation_df column order drifted from the assignment order above: "
        f"{list(points_validation_df.columns)} != {expected_columns}"
    )
    points_validation_df.to_parquet(out_dir / "points_validation.parquet", index=False)
    print(f"Saved {len(points_validation_df)} row(s) to {out_dir / 'points_validation.parquet'}")

    return points_validation_df, swot_df


def main():
    parser = argparse.ArgumentParser(
        description="Build the SWOT-anchored points_validation table for a validation config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a configs/validation/*.yaml file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = VALIDATION_RESULTS_DIR / config["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(config["start_date"], config["end_date"], freq="1D")]
    print(f"Validation config '{config['name']}': {len(target_dates)} target day(s) ({config['start_date']} to {config['end_date']}).")

    drive_base, _ = get_drive_base_path()

    print(f"Loading SWOT points for '{config['swot_config']}'...")
    swot_df = _load_swot_points(config["swot_config"], drive_base)

    print(f"Loading DUACS config '{config['duacs_config']}'...")
    duacs_config = _find_config_by_name(CONFIGS_DIR / "data_l4", config["duacs_config"])
    duacs_ds = load_l4_dataset(duacs_config)

    print("Stage 1: building grid snapshots (DUACS + every drf_run, over target_dates)...")
    duacs_snapshots = save_duacs_grid_snapshots(config, target_dates, duacs_ds, out_dir)
    drf_snapshots = build_grid_snapshots(config, target_dates, out_dir)

    duacs_name = config["duacs_config"]
    grid_snapshots = {(duacs_name, date_str): (mean, None) for date_str, mean in duacs_snapshots.items()}
    grid_snapshots.update(drf_snapshots)

    print("Stage 2: building points_validation.parquet from grid_snapshots...")
    points_validation_df, swot_df_filtered = build_points_table(config, swot_df, grid_snapshots, out_dir)

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
