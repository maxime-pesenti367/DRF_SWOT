"""
Regenerates the two "how did training go" plots for an exp5 run from the
CSVs experiment_5.py already saves -- no retraining or re-running the
Bayesian-optimization search needed.

  - search_progress.png  <- search_history.csv   (every BO candidate tried)
  - training_curve.png   <- training_curve.csv    (final ensemble, per epoch)

Usage:
    python plot_search_history.py --results-dir results/<config-name>

Also directly importable (plot_search_progress / plot_training_curve) --
experiment_5.py itself calls these two functions inline at the end of a run,
so this module is both the standalone regeneration tool (mirroring
replot_grid.py's relationship to the whole-globe grid snapshot) and the
single source of truth for the plotting logic, not a separate copy of it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def _needs_wide_range_scale(values, ratio_threshold=20):
    """Whether a column's actual values in this run span a wide enough
    range to need a log-type y-axis. Deciding this per-run from the real
    data (rather than hardcoding it per-column) matters: final_loss/
    val_nlpd/train_loss can span many orders of magnitude in one run and
    be tightly clustered in another (e.g. a 2-round search where both
    val_nlpd values land in the 500s) -- log-decade tick placement can
    leave the axis with zero labeled ticks when the whole range sits
    inside one decade, so forcing a log scale on a narrow range makes the
    plot worse, not better."""
    abs_vals = values.abs()
    nonzero = abs_vals[abs_vals > 1e-12]
    if len(nonzero) < 2:
        return False
    return (nonzero.max() / nonzero.min()) > ratio_threshold


def plot_search_progress(results_dir):
    """Plots final_loss (the actual BO objective), val_rmse, val_nlpd, and
    val_crps against BO round, distinguishing the random initial_samples
    from the GP-guided iterations, with the winning round marked."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "search_history.csv")

    initial = df[df["round_type"] == "initial_sample"]
    iteration = df[df["round_type"] == "iteration"]
    winner = df[df["is_winner"]]

    panels = [
        ("final_loss", "final_loss (BO objective)"),
        ("val_rmse", "val RMSE"),
        ("val_nlpd", "val NLPD"),
        ("val_crps", "val CRPS"),
    ]
    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(8, 11))

    for ax, (col, ylabel) in zip(axes, panels):
        ax.scatter(
            initial["round"], initial[col],
            color="tab:gray", label="initial_samples", zorder=2,
        )
        if len(iteration) > 0:
            ax.plot(
                iteration["round"], iteration[col], "o-",
                color="tab:blue", label="iterations", zorder=2,
            )
        ax.scatter(
            winner["round"], winner[col],
            color="tab:red", marker="*", s=200, label="winner", zorder=3,
        )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        # final_loss and val_nlpd can both span several orders of magnitude
        # in one run (and val_nlpd can go negative for a well-calibrated
        # round) -- symlog keeps small, well-behaved values readable
        # instead of every point but one being squashed flat near zero/one
        # dominant outlier. But only switch to it when *this run's* values
        # actually warrant it -- see _needs_symlog.
        if _needs_wide_range_scale(df[col]):
            ax.set_yscale("symlog")

    axes[0].legend()
    axes[-1].set_xlabel("BO round")
    fig.suptitle("Bayesian-optimization search progress")
    fig.tight_layout()
    fig.savefig(results_dir / "search_progress.png", dpi=200)
    plt.close(fig)
    print(f"Saved {results_dir / 'search_progress.png'}")


def plot_training_curve(results_dir):
    """Plots the final (checkpointed) ensemble's own per-epoch train/val
    Huber loss and val RMSE, one line per ensemble seed."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "training_curve.csv")

    fig, (ax_loss, ax_rmse) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    for seed, group in df.groupby("seed"):
        color = f"C{seed % 10}"
        # Markers matter here, not just line style: num_epochs=1 is common
        # in these configs, and a lone point with a line-only style (no
        # marker) renders nothing at all -- this previously made the whole
        # Huber-loss panel silently blank for any 1-epoch run.
        ax_loss.plot(group["epoch"], group["train_loss"], "-o", color=color, label=f"seed {seed} train")
        ax_loss.plot(group["epoch"], group["val_loss"], "--s", color=color, label=f"seed {seed} val")
        ax_rmse.plot(group["epoch"], group["val_rmse"], "-o", color=color, label=f"seed {seed}")

    ax_loss.set_ylabel("Huber loss")
    ax_loss.legend(fontsize=7, ncol=2)
    ax_loss.grid(alpha=0.3)
    # Huber loss and RMSE are both non-negative, so a plain log scale (not
    # symlog) works here -- unlike search_progress's val_nlpd, which can go
    # negative. Late-training epochs often converge to losses clustered
    # near zero across seeds, which a linear axis flattens into an
    # indistinguishable line; log scale keeps that separation visible.
    if _needs_wide_range_scale(pd.concat([df["train_loss"], df["val_loss"]])):
        ax_loss.set_yscale("log")
    ax_rmse.set_ylabel("val RMSE")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.grid(alpha=0.3)
    if _needs_wide_range_scale(df["val_rmse"]):
        ax_rmse.set_yscale("log")
    fig.suptitle("Final ensemble training curve (per seed)")
    fig.tight_layout()
    fig.savefig(results_dir / "training_curve.png", dpi=200)
    plt.close(fig)
    print(f"Saved {results_dir / 'training_curve.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate search-progress and training-curve plots from saved exp5 CSVs"
    )
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Path to a results/<config-name> folder containing search_history.csv / training_curve.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = SCRIPT_DIR / results_dir

    plot_search_progress(results_dir)
    plot_training_curve(results_dir)