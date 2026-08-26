"""
Runs experiment_5.py across multiple configs back-to-back, and optionally
generates those configs itself from a sweep spec first.

Three ways to pick which configs to run (exactly one required):

    --configs <path-or-glob> [<path-or-glob> ...]
        Explicit config paths and/or glob patterns, relative to this
        directory. Run in the order given.

    --configs-file <path>
        A text file with one config path/glob per line ('#' comments and
        blank lines ignored) -- for a reusable, named batch you don't want
        to retype on the command line every time.

    --sweep <path>
        A sweep-spec YAML (see configs/batches/example_seed_sweep.yaml):
        one base_config plus a list of override dicts. Each override is
        applied on top of a fresh copy of the base config (dotted keys,
        e.g. training.learning_rate) and written out as its own config
        file under configs/exp5/generated/<sweep-spec-stem>/, auto-named
        from the overridden keys/values unless the override sets its own
        `name`. All N generated configs are then run.
        Note: since this round-trips the base config through
        yaml.safe_load/safe_dump, generated files lose the original's
        comments and exact number formatting (e.g. 1e-5 -> 1.0e-05) --
        cosmetic only, doesn't change what's actually run.

Runs are strictly sequential -- one experiment_5.py process runs to
completion (and exits, freeing its CUDA context) before the next starts.
experiment_5.py's own Bayesian-optimization search already runs several
ensemble members concurrently within a single run (see
training.max_parallel_models), sized against a single run's GPU budget;
running multiple experiment_5.py processes at once would contend for the
same GPU memory on top of that, so this runner never does.

Each run's combined stdout/stderr streams live to the console (same as
running experiment_5.py directly) and is also duplicated to
results/_batch_logs/<batch-timestamp>/<config-stem>.log. A failed run
(non-zero exit code) is recorded and the batch continues on to the next
config by default -- pass --stop-on-failure to abort the whole batch
instead. A one-line summary table prints at the end, followed by
aggregate_results.py (skip with --no-aggregate) so results.csv from every
successful run in the batch -- and every earlier run -- lands in
results/all_results_summary.csv.

Usage:
    python run_batch.py --configs configs/exp5/exp_a.yaml configs/exp5/exp_b.yaml
    python run_batch.py --configs "configs/exp5/exp_my_all_sats_4_weeks_*.yaml"
    python run_batch.py --configs-file configs/batches/my_batch.txt
    python run_batch.py --sweep configs/batches/example_seed_sweep.yaml
    python run_batch.py --sweep configs/batches/example_seed_sweep.yaml --stop-on-failure
"""

import argparse
import copy
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_config_args(patterns, script_dir):
    paths = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(script_dir.glob(pattern))
            if not matches:
                print(f"WARNING: pattern {pattern!r} matched no files -- skipping")
            paths.extend(matches)
        else:
            p = Path(pattern)
            paths.append(p if p.is_absolute() else script_dir / p)
    return paths


def _read_configs_file(list_path, script_dir):
    lines = [
        line.strip()
        for line in list_path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return _resolve_config_args(lines, script_dir)


def _set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _slugify_value(value):
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", str(value))


def _override_slug(overrides):
    parts = [
        f"{dotted_key.split('.')[-1]}{_slugify_value(value)}"
        for dotted_key, value in overrides.items()
        if dotted_key != "name"
    ]
    return "_".join(parts)


def generate_sweep_configs(sweep_path, script_dir):
    with open(sweep_path) as f:
        sweep_spec = yaml.safe_load(f)

    base_config_path = script_dir / sweep_spec["base_config"]
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    out_dir = script_dir / "configs" / "exp5" / "generated" / sweep_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    base_stem = base_config_path.stem
    generated_paths = []
    used_names = set()
    for i, overrides in enumerate(sweep_spec["overrides"]):
        config = copy.deepcopy(base_config)
        for dotted_key, value in overrides.items():
            if dotted_key != "name":
                _set_nested(config, dotted_key, value)

        explicit_name = overrides.get("name")
        if explicit_name:
            config_name = explicit_name
        else:
            slug = _override_slug(overrides)
            config_name = f"{base_stem}__{slug}" if slug else f"{base_stem}__variant{i}"

        if config_name in used_names:
            raise ValueError(
                f"Duplicate generated config name {config_name!r} in {sweep_path} -- "
                "give one of the colliding overrides an explicit 'name' to disambiguate"
            )
        used_names.add(config_name)

        out_path = out_dir / f"{config_name}.yaml"
        with open(out_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Generated {out_path.relative_to(script_dir)}")
        generated_paths.append(out_path)

    return generated_paths


def run_one(python_exe, script_dir, config_path, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python_exe, "experiment_5.py", "--config", str(config_path)]
    print(f"\n{'=' * 80}\nRunning: {' '.join(cmd)}\nLog: {log_path}\n{'=' * 80}")

    start = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd, cwd=script_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
        returncode = proc.wait()
    elapsed = time.monotonic() - start
    return returncode == 0, elapsed, returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", help="Explicit config paths and/or glob patterns")
    parser.add_argument("--configs-file", type=str, help="Text file listing one config path/glob per line")
    parser.add_argument("--sweep", type=str, help="Sweep-spec YAML: base_config + override list")
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Abort the batch on the first failed run instead of continuing",
    )
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Skip auto-running aggregate_results.py after the batch finishes",
    )
    args = parser.parse_args()

    if sum(bool(x) for x in (args.configs, args.configs_file, args.sweep)) != 1:
        parser.error("Pass exactly one of --configs, --configs-file, or --sweep")

    if args.sweep:
        config_paths = generate_sweep_configs(SCRIPT_DIR / args.sweep, SCRIPT_DIR)
    elif args.configs_file:
        config_paths = _read_configs_file(SCRIPT_DIR / args.configs_file, SCRIPT_DIR)
    else:
        config_paths = _resolve_config_args(args.configs, SCRIPT_DIR)

    if not config_paths:
        parser.error("No config files resolved -- nothing to run")

    batch_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = SCRIPT_DIR / "results" / "_batch_logs" / batch_id
    print(f"Batch of {len(config_paths)} run(s). Logs -> {log_dir.relative_to(SCRIPT_DIR)}")

    results = []
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{len(config_paths)}] {config_path.name}")
        log_path = log_dir / f"{config_path.stem}.log"
        success, elapsed, returncode = run_one(sys.executable, SCRIPT_DIR, config_path, log_path)
        results.append({
            "config": config_path.name, "success": success,
            "elapsed_min": elapsed / 60, "returncode": returncode,
        })
        status = "OK" if success else f"FAILED (exit {returncode})"
        print(f"[{i}/{len(config_paths)}] {config_path.name}: {status} in {elapsed / 60:.1f} min")
        if not success and args.stop_on_failure:
            print("Stopping batch (--stop-on-failure).")
            break

    print(f"\n{'=' * 80}\nBatch summary\n{'=' * 80}")
    for r in results:
        status = "OK" if r["success"] else f"FAILED (exit {r['returncode']})"
        print(f"  {r['config']:<65} {status:<18} {r['elapsed_min']:.1f} min")

    n_failed = sum(1 for r in results if not r["success"])
    n_skipped = len(config_paths) - len(results)
    print(f"\n{len(results) - n_failed}/{len(config_paths)} succeeded.", end="")
    if n_skipped:
        print(f" {n_skipped} skipped (stopped early).", end="")
    print(f"\nLogs: {log_dir}")

    if not args.no_aggregate and any(r["success"] for r in results):
        print("\nRunning aggregate_results.py...")
        subprocess.run([sys.executable, "aggregate_results.py"], cwd=SCRIPT_DIR)

    sys.exit(1 if (n_failed or n_skipped) else 0)


if __name__ == "__main__":
    main()
