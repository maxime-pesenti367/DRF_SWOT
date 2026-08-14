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


def plot_search_progress(results_dir):
    """Plots final_loss (the actual BO objective) and val_rmse (the more
    interpretable accuracy proxy) against BO round, distinguishing the
    random initial_samples from the GP-guided iterations, with the winning
    round marked."""
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "search_history.csv")

    initial = df[df["round_type"] == "initial_sample"]
    iteration = df[df["round_type"] == "iteration"]
    winner = df[df["is_winner"]]

    fig, (ax_loss, ax_rmse) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    for ax, col, ylabel in [
        (ax_loss, "final_loss", "final_loss (BO objective)"),
        (ax_rmse, "val_rmse", "val RMSE"),
    ]:
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

    # final_loss can span several orders of magnitude in a single search
    # (e.g. a candidate near the pathological end of the spatial_lengthscale
    # bound can spike the regularization term into the tens of thousands) --
    # symlog keeps small, well-behaved values readable instead of every
    # point but one being squashed flat near zero. val_rmse doesn't need
    # this; it stays comfortably bounded.
    ax_loss.set_yscale("symlog")
    ax_loss.legend()
    ax_rmse.set_xlabel("BO round")
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
        ax_loss.plot(group["epoch"], group["train_loss"], "-", color=color, label=f"seed {seed} train")
        ax_loss.plot(group["epoch"], group["val_loss"], "--", color=color, label=f"seed {seed} val")
        ax_rmse.plot(group["epoch"], group["val_rmse"], "-o", color=color, label=f"seed {seed}")

    ax_loss.set_ylabel("Huber loss")
    ax_loss.legend(fontsize=7, ncol=2)
    ax_loss.grid(alpha=0.3)
    ax_rmse.set_ylabel("val RMSE")
    ax_rmse.set_xlabel("epoch")
    ax_rmse.grid(alpha=0.3)
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