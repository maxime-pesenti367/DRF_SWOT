"""
Aggregates every exp5 run's results.csv (results/<config-name>/results.csv)
into one summary CSV for eyeballing which hyperparameters tend to produce
good/bad models across experiments.

results.csv's schema has changed twice as exp5 grew features, so this reads
whatever columns are actually present per-file rather than assuming one
fixed schema:
  - oldest runs: one row, no `selection_criterion`, no bias/variance columns.
  - mid runs: one row, adds `test_bias`/`test_variance_of_diff`.
  - current runs: one or two rows (`selection_criterion` =
    final_loss_winner/val_rmse_winner, or final_loss_and_val_rmse_winner
    when they coincide), adds `search_val_rmse`.
Missing columns for a given row are left blank in the output rather than
raising, so all three eras can sit in one table.

exp4/results.csv is a different, incompatible format entirely (raw
PyTorch-tensor param string, NLL only, no test_rmse) -- excluded here since
it can't be ranked by test_rmse alongside the exp5 runs; see CLAUDE.md.

Usage:
    python aggregate_results.py
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "all_experiments_summary.csv"

COLUMNS = [
    "experiment_name",
    "winner",
    "spatial_lengthscale",
    "temporal_lengthscale",
    "amplitude",
    "lengthscale2",
    "amplitude2",
    "test_rmse",
    "test_nlpd",
    "test_crps",
    "test_bias",
    "test_variance_of_diff",
    "best_val_loss",
    "search_val_rmse",
]


def main():
    csv_paths = sorted(RESULTS_DIR.glob("*/results.csv"))

    rows = []
    skipped = []
    for csv_path in csv_paths:
        experiment_name = csv_path.parent.name
        df = pd.read_csv(csv_path)

        if "test_rmse" not in df.columns:
            skipped.append(experiment_name)
            continue

        for _, row in df.iterrows():
            rows.append({
                "experiment_name": experiment_name,
                "winner": row.get("selection_criterion", ""),
                **{col: row.get(col, "") for col in COLUMNS if col not in ("experiment_name", "winner")},
            })

    summary_df = pd.DataFrame(rows, columns=COLUMNS)
    summary_df = summary_df.sort_values("test_rmse", ascending=False)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Aggregated {len(rows)} row(s) from {len(csv_paths) - len(skipped)} experiment(s).")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible results.csv (no test_rmse column): {skipped}")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
