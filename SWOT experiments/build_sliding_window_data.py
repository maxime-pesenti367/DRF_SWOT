"""
Builds one pre-split, pre-normalized .pt tensor set (+ overview plot) per
target day described by a data_sliding_window config -- the sliding-window
analog of build_experiment_data.py, which this script calls into directly
(same fetch venv, no subprocess/CLI boundary) rather than duplicating its
fetch/split/tensorize logic.

For each target day T in [start_date, end_date] (stepped by step_days),
fetches a window of sliding_day_range days centered on T's 00:00 -- i.e.
[T - sliding_day_range/2, T + sliding_day_range/2] -- independently of
every other target day. Deliberately NOT a slice of one larger fetch: each
day's normalization stats and split are computed from just that day's own
narrow window (see build_experiment_data.py's module docstring for why --
temporal normalization is the train split's own mean/std, which would be
wrong if silently reused across a wider window than what was actually
fetched for that day). A day's window can extend outside the master
config's own start_date/end_date -- e.g. the first target day still needs
real data from one half-window before it -- since start_date/end_date here
describe the target *prediction* days, not the fetch range.

Each day's generated data config is written to
configs/data_sliding_window/generated/<name>/<date>.yaml -- gitignored,
regenerated on demand, since it's fully and deterministically reconstructible
from the master config alone (nothing is lost by not tracking it). Its
`name` is "<master name>/<date>" (e.g. "july25_2daywindow/2025-07-01"),
which -- via plain pathlib string-joining already in
build_experiment_data.py/copernicus_pipeline.py, no special-casing needed
anywhere -- lands its tensors/plot at pytorch tensors/<name>/<date>/ on
Drive.

Skips a day whose split .pt file(s) already exist on Drive, so a run that
fails partway through (e.g. day 16 of 31) can simply be re-run and will
pick up where it left off -- mirrors copernicus_l4_pipeline.py's existing
skip-if-exists convention. Pass --force to rebuild every day regardless.

Usage:
    python build_sliding_window_data.py --config configs/data_sliding_window/july25_2daywindow.yaml
    python build_sliding_window_data.py --config configs/data_sliding_window/july25_2daywindow.yaml --force
"""

# Forces the non-interactive Agg backend before matplotlib.pyplot gets
# imported by anything below (transitively, via display_tracks.py) -- must
# happen first, backend switching after pyplot's already been imported is
# unreliable. This script -- unlike the one-shot build_experiment_data.py/
# build_l4_data.py CLI invocations -- runs a single long-lived process
# through up to 31 fetch+plot cycles; matplotlib's default backend on this
# machine pulls in tkinter, and a Tk-backed figure/image object finalized
# later (e.g. by the garbage collector, possibly off the main thread) can
# crash with "RuntimeError: main thread is not in main loop" -- observed
# live during the first real run of this script. Agg never creates any
# Tk/GUI objects at all, which sidesteps the whole class of issue -- exactly
# right here since save_path is always passed (never plt.show()).
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

import pandas as pd
import yaml

from build_experiment_data import build_experiment_data
from copernicus_pipeline import get_drive_base_path
from sliding_window_utils import generate_target_dates

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_DATA_CONFIG_DIR = SCRIPT_DIR / "configs" / "data_sliding_window" / "generated"


def generate_day_configs(master_config):
    """Yields one (date_str, day_config) pair per target day. Target dates
    themselves come from generate_target_dates() (shared with
    run_sliding_window.py); sliding_day_range -- turning a target day into
    an actual fetch window -- is fetch-specific and stays local here."""
    half_window = pd.Timedelta(days=master_config["sliding_day_range"] / 2)

    for date_str in generate_target_dates(master_config):
        target_day = pd.Timestamp(date_str)
        day_config = {
            "name": f"{master_config['name']}/{date_str}",
            "product_type": master_config["product_type"],
            "satellites": master_config["satellites"],
            "freq": master_config["freq"],
            "start_date": (target_day - half_window).strftime("%Y-%m-%d"),
            "end_date": (target_day + half_window).strftime("%Y-%m-%d"),
            "variables": master_config["variables"],
            "splits": master_config["splits"],
        }
        yield date_str, day_config


def day_already_built(day_config, drive_base):
    """Whether every split this day config lists already has a saved .pt
    file on Drive -- checked against the actual final artifact, not just
    the day's folder existing (which mkdir(parents=True) creates early,
    before any of the expensive fetch/tensorize work happens), so a day
    that crashed partway through is correctly retried, not skipped."""
    for split_config in day_config["splits"]:
        pt_path = drive_base / "pytorch tensors" / day_config["name"] / f"{split_config['name']}.pt"
        if not pt_path.exists():
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build one dataset per target day in a sliding-window data config")
    parser.add_argument("--config", type=str, required=True, help="Path to a data_sliding_window config YAML file")
    parser.add_argument("--no-plot", action="store_false", dest="plot", help="Skip the per-day tracks overview PNG")
    parser.add_argument("--force", action="store_true", help="Rebuild every day even if already built")
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    with open(args.config) as f:
        master_config = yaml.safe_load(f)

    generated_dir = GENERATED_DATA_CONFIG_DIR / master_config["name"]
    generated_dir.mkdir(parents=True, exist_ok=True)

    drive_base, _ = get_drive_base_path()

    day_configs = list(generate_day_configs(master_config))
    print(f"Sliding-window config '{master_config['name']}': {len(day_configs)} target day(s).")

    failed_days = []
    for i, (date_str, day_config) in enumerate(day_configs, 1):
        day_config_path = generated_dir / f"{date_str}.yaml"
        with open(day_config_path, "w") as f:
            yaml.safe_dump(day_config, f, sort_keys=False)

        if not args.force and day_already_built(day_config, drive_base):
            print(f"[{i}/{len(day_configs)}] {date_str}: already built, skipping.")
            continue

        print(
            f"[{i}/{len(day_configs)}] {date_str}: building "
            f"(window {day_config['start_date']} to {day_config['end_date']})..."
        )
        # One day's fetch failing (e.g. a transient Copernicus API hiccup, or
        # a Windows file-lock race in zarr's async writer -- both observed
        # live) shouldn't take down the other ~30 days in the same run.
        # Left unbuilt, a failed day is picked up correctly on the next
        # invocation by the skip-if-exists check above -- no special
        # handling needed beyond not crashing here.
        try:
            build_experiment_data(day_config, plot=args.plot)
        except Exception as e:
            print(f"[{i}/{len(day_configs)}] {date_str}: FAILED -- {e!r}")
            failed_days.append(date_str)

    print(f"\nDone. {len(day_configs) - len(failed_days)}/{len(day_configs)} day(s) built.")
    if failed_days:
        print(f"Failed day(s) (re-run this script to retry them): {failed_days}")


if __name__ == "__main__":
    main()
