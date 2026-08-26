"""
Aggregates every exp5 run's results.csv and search_history.csv across
results/<config-name>/ into two summary CSVs, plus one pooled scatter-grid
PNG for eyeballing whether individual hyperparameters have any general
relationship to performance across many different experiments.

results.csv's schema has changed twice as exp5 grew features, so
aggregate_results() reads whatever columns are actually present per-file
rather than assuming one fixed schema:
  - oldest runs: one row, no `selection_criterion`, no bias/variance columns.
  - mid runs: one row, adds `test_bias`/`test_variance_of_diff`.
  - current runs: one or two rows (`selection_criterion` =
    final_loss_winner/val_rmse_winner, or final_loss_and_val_rmse_winner
    when they coincide), adds `search_val_rmse`.
Missing columns for a given row are left blank in the output rather than
raising, so all three eras can sit in one table. all_results_summary.csv is
the renamed all_experiments_summary.csv -- kept paired by name with the
results.csv it aggregates, now that all_search_history.csv (aggregating
search_history.csv) sits alongside it.

search_history.csv's schema is simpler to aggregate: every era already has
the 5 hyperparameters, final_loss, val_rmse and val_nlpd; only
val_bias/val_variance_of_diff/is_val_rmse_winner (SWOT-optimizer-path-only,
see spherical_uq_methods_SWOT.py) vary by run. aggregate_search_history()
just concatenates every file (pandas fills missing columns with NaN), no
per-column fallback needed the way aggregate_results() requires.

exp4/results.csv is a different, incompatible format entirely (raw
PyTorch-tensor param string, NLL only, no test_rmse) -- excluded here since
it can't be ranked by test_rmse alongside the exp5 runs; see CLAUDE.md.

plot_hyperparameter_grid() pools every aggregated search_history.csv row
(every BO trial from every experiment) into one scatter-grid PNG: one panel
per (hyperparameter, metric) pair -- 5 hyperparameters x 3 metrics
(final_loss, val_rmse, val_nlpd). This is deliberately NOT a rigorous
cross-experiment comparison (different experiments train on different
data/depths/search budgets, so pooling their trials is an apples-to-oranges
mix) -- it's purely for eyeballing whether a hyperparameter's value has any
*general* pattern with performance, independent of which experiment it came
from. x is always log-scaled (every hyperparameter's BO search bounds span
several orders of magnitude); y-scale (log/symlog/linear) is decided per
metric from the real pooled data via plot_search_history.py's own
_needs_wide_range_scale, reused here rather than re-implemented, matching
that module's existing per-run search-progress plot. x is shared within
each row (same hyperparameter across all 3 metric columns) and y is shared
within each column (same metric across all 5 hyperparameter rows) --
deliberately never shared across a row/column boundary, since that would
put differently-scaled measures on one axis (the #1 chart-design mistake --
see e.g. final_loss and val_rmse, which differ by orders of magnitude).

Usage:
    python aggregate_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_search_history import _needs_wide_range_scale

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_SUMMARY_OUTPUT_PATH = RESULTS_DIR / "all_results_summary.csv"
SEARCH_HISTORY_OUTPUT_PATH = RESULTS_DIR / "all_search_history.csv"
HYPERPARAMETER_GRID_OUTPUT_PATH = RESULTS_DIR / "hyperparameter_performance_grid.png"

RESULTS_SUMMARY_COLUMNS = [
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

# Row order == hyperparameter row order in the grid PNG; column order ==
# metric column order there too.
HYPERPARAMETERS = [
    "spatial_lengthscale",
    "temporal_lengthscale",
    "amplitude",
    "lengthscale2",
    "amplitude2",
]
METRICS = ["final_loss", "val_rmse", "val_nlpd"]


def aggregate_results():
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
                **{col: row.get(col, "") for col in RESULTS_SUMMARY_COLUMNS if col not in ("experiment_name", "winner")},
            })

    summary_df = pd.DataFrame(rows, columns=RESULTS_SUMMARY_COLUMNS)
    summary_df = summary_df.sort_values("test_rmse", ascending=True)
    summary_df.to_csv(RESULTS_SUMMARY_OUTPUT_PATH, index=False)

    print(f"Aggregated {len(rows)} row(s) from {len(csv_paths) - len(skipped)} experiment(s).")
    if skipped:
        print(f"Skipped {len(skipped)} incompatible results.csv (no test_rmse column): {skipped}")
    print(f"Saved {RESULTS_SUMMARY_OUTPUT_PATH}")


def aggregate_search_history():
    csv_paths = sorted(RESULTS_DIR.glob("*/search_history.csv"))

    dfs = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.insert(0, "experiment_name", csv_path.parent.name)
        dfs.append(df)

    if not dfs:
        print("No search_history.csv files found -- skipping search-history aggregation.")
        return pd.DataFrame()

    # sort=False keeps each source file's own column order rather than
    # alphabetizing; missing columns (val_bias/val_variance_of_diff/
    # is_val_rmse_winner, absent on pre-SWOT-optimizer runs) are filled NaN.
    history_df = pd.concat(dfs, ignore_index=True, sort=False)
    history_df = history_df.sort_values("val_rmse", ascending=True)
    history_df.to_csv(SEARCH_HISTORY_OUTPUT_PATH, index=False)

    print(f"Aggregated {len(history_df)} row(s) from {len(csv_paths)} experiment(s).")
    print(f"Saved {SEARCH_HISTORY_OUTPUT_PATH}")
    return history_df


def plot_hyperparameter_grid(history_df):
    if history_df.empty:
        print("No search-history data to plot -- skipping hyperparameter grid.")
        return

    fig, axes = plt.subplots(
        len(HYPERPARAMETERS), len(METRICS),
        figsize=(4 * len(METRICS), 3 * len(HYPERPARAMETERS)),
        sharex="row", sharey="col",
    )

    for row_idx, hyperparam in enumerate(HYPERPARAMETERS):
        x = history_df[hyperparam]
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            y = history_df[metric]
            ax.scatter(x, y, s=10, alpha=0.2, color="tab:blue", edgecolors="none")
            ax.set_xscale("log")
            # Decided per metric from the real pooled data, same heuristic
            # plot_search_history.py's own per-run plot uses -- not
            # hardcoded, since e.g. val_rmse might not need it while
            # final_loss/val_nlpd usually will (pooling ~25 experiments
            # trained on very different datasets/scales makes final_loss
            # span *dozens* of orders of magnitude, far wider than any
            # single run's own search history). Plain "log" (not "symlog")
            # whenever every pooled value is positive -- matches
            # plot_training_curve()'s existing reasoning that Huber
            # loss/RMSE don't need symlog's negative-value handling, only
            # val_nlpd (which can go negative for a well-calibrated round)
            # does. This also sidesteps a real matplotlib usability gap:
            # SymmetricalLogLocator doesn't thin ticks to one-per-decade the
            # way LogLocator does, so symlog over a 27-decade span like
            # pooled final_loss renders as illegible stacked tick labels.
            if _needs_wide_range_scale(y):
                ax.set_yscale("log" if (y > 0).all() else "symlog")
            ax.grid(alpha=0.3)
            ax.set_xlabel(hyperparam)
            ax.set_ylabel(metric)
            if row_idx == 0:
                ax.set_title(metric)

    n_experiments = history_df["experiment_name"].nunique()
    fig.suptitle(f"Pooled Bayesian-optimization trials across {n_experiments} experiment(s) (for eyeballing only)")
    fig.tight_layout()
    fig.savefig(HYPERPARAMETER_GRID_OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {HYPERPARAMETER_GRID_OUTPUT_PATH}")


def main():
    aggregate_results()
    history_df = aggregate_search_history()
    plot_hyperparameter_grid(history_df)


if __name__ == "__main__":
    main()
