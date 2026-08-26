"""
Trains one experiment_5.py model per target day described by an
exp_sliding_window config -- the training-side analog of
build_sliding_window_data.py, run in the main/train venv (often on a
different, remote/GPU machine from wherever build_sliding_window_data.py
ran).

An exp_sliding_window master config (configs/exp_sliding_window/*.yaml)
looks exactly like a normal exp5 config -- data/device/seed/model/
bayesian_optimization/uq_method/training -- except its data.experiment_name
names the sliding-window *family* (e.g. "july25_2daywindow"), not one
specific dataset. This script reads
configs/data_sliding_window/<that name>.yaml (a small, git-synced text
file -- doesn't need the fetch venv's dependencies, see
sliding_window_utils.py) to recompute the exact same target-day sequence
build_sliding_window_data.py used, then for each day:

  1. Checks whether that day's tensors have actually synced to this machine
     yet -- rclone is a manual step in this project's workflow, so "not
     synced yet" is a normal, expected state to skip past, not an error.
  2. Checks whether that day is already trained (results/<master config
     stem>/<date>/results.csv already exists) -- skips if so, unless
     --force.
  3. Otherwise clones every setting from the master (device/seed/model/
     bayesian_optimization/uq_method/training) into a fresh per-day config,
     substitutes that day's own data.experiment_name/results_name, and runs
     experiment_5.py against it as its own subprocess -- deliberately not
     an in-process call. Each day's training has its own GPU/CUDA
     multiprocessing pool; a fresh subprocess per day guarantees a clean
     GPU state for each one rather than accumulating state across 31 days
     in one long-lived parent process.

Whichever candidate(s) experiment_5.py itself decides to retrain for a
given day (final_loss_winner / val_rmse_winner / both collapsed into one
when they coincide) is left entirely to its own existing, unmodified logic
-- this script never restricts or second-guesses that per day.

Generated per-day configs land in
configs/exp_sliding_window/generated/<master-config-stem>/<date>.yaml --
gitignored, regenerated on demand (see build_sliding_window_data.py's
module docstring for why that's fine: fully reproducible from the master
config alone, nothing lost by not tracking them).

Usage:
    python run_sliding_window.py --config configs/exp_sliding_window/exp_july25_2daywindow.yaml
    python run_sliding_window.py --config configs/exp_sliding_window/exp_july25_2daywindow.yaml --force
"""

import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import yaml

from drive_paths import get_drive_base_path
from sliding_window_utils import generate_target_dates

SCRIPT_DIR = Path(__file__).resolve().parent
GENERATED_EXP_CONFIG_DIR = SCRIPT_DIR / "configs" / "exp_sliding_window" / "generated"

# Cloned verbatim from the master into every day's config; "data" is handled
# separately since it's the one thing that varies per day.
CLONED_KEYS = ["device", "seed", "model", "bayesian_optimization", "uq_method", "training"]


def build_day_config(master_config, master_stem, data_name, split_name, date_str):
    day_config = {key: copy.deepcopy(master_config[key]) for key in CLONED_KEYS if key in master_config}
    day_config["results_name"] = f"{master_stem}/{date_str}"
    day_config["data"] = {"experiment_name": f"{data_name}/{date_str}", "split_name": split_name}
    return day_config


def day_data_ready(data_name, split_name, date_str, drive_base):
    return (drive_base / "pytorch tensors" / data_name / date_str / f"{split_name}.pt").exists()


def day_already_trained(master_stem, date_str):
    return (SCRIPT_DIR / "results" / master_stem / date_str / "results.csv").exists()


def run_one_day(day_config_path, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "experiment_5.py", "--config", str(day_config_path)]
    print(f"\n{'=' * 80}\nRunning: {' '.join(cmd)}\nLog: {log_path}\n{'=' * 80}")

    start = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        returncode = proc.wait()
    elapsed = time.monotonic() - start
    return returncode == 0, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Train one experiment_5.py model per target day in a sliding-window exp config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to an exp_sliding_window config YAML file")
    parser.add_argument("--force", action="store_true", help="Retrain every day even if already trained")
    args = parser.parse_args()

    config_path = Path(args.config)
    master_stem = config_path.stem
    with open(config_path) as f:
        master_config = yaml.safe_load(f)

    data_name = master_config["data"]["experiment_name"]
    split_name = master_config["data"]["split_name"]

    with open(SCRIPT_DIR / "configs" / "data_sliding_window" / f"{data_name}.yaml") as f:
        data_master_config = yaml.safe_load(f)

    target_dates = generate_target_dates(data_master_config)
    drive_base, _ = get_drive_base_path()
    generated_dir = GENERATED_EXP_CONFIG_DIR / master_stem
    generated_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sliding-window exp config '{master_stem}': {len(target_dates)} target day(s).")

    results = []
    for i, date_str in enumerate(target_dates, 1):
        if not day_data_ready(data_name, split_name, date_str, drive_base):
            print(f"[{i}/{len(target_dates)}] {date_str}: data not yet synced, skipping.")
            continue

        if not args.force and day_already_trained(master_stem, date_str):
            print(f"[{i}/{len(target_dates)}] {date_str}: already trained, skipping.")
            continue

        day_config = build_day_config(master_config, master_stem, data_name, split_name, date_str)
        day_config_path = generated_dir / f"{date_str}.yaml"
        with open(day_config_path, "w") as f:
            yaml.safe_dump(day_config, f, sort_keys=False)

        print(f"[{i}/{len(target_dates)}] {date_str}: training...")
        log_path = SCRIPT_DIR / "results" / master_stem / "_logs" / f"{date_str}.log"
        success, elapsed = run_one_day(day_config_path, log_path)
        results.append((date_str, success, elapsed))
        status = "OK" if success else "FAILED"
        print(f"[{i}/{len(target_dates)}] {date_str}: {status} in {elapsed / 60:.1f} min")

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    for date_str, success, elapsed in results:
        print(f"  {date_str}: {'OK' if success else 'FAILED'} ({elapsed / 60:.1f} min)")
    n_failed = sum(1 for _, success, _ in results if not success)
    n_ready = sum(1 for i, d in enumerate(target_dates) if day_data_ready(data_name, split_name, d, drive_base))
    print(f"\n{len(results) - n_failed}/{len(results)} attempted day(s) succeeded.")
    print(f"{n_ready}/{len(target_dates)} target day(s) have synced data so far.")


if __name__ == "__main__":
    main()
