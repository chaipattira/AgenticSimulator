"""
plot_efficiency.py — sample-efficiency curves for paper Figure 1.

Usage:
    python analysis/plot_efficiency.py --summary benchmark_results/benchmark_summary.csv
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"agent": "#e41a1c", "random": "#aaaaaa",
          "botorch": "#377eb8", "optuna": "#4daf4a"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("analysis/sample_efficiency.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for method in methods:
        subset = df[df["method"] == method]
        converged = subset[subset["converged"]]
        calls = converged["n_calls"].values if len(converged) > 0 else []
        ax.hist(calls, bins=10, alpha=0.6,
                label=f"{method} ({len(converged)}/{len(subset)} converged)",
                color=COLORS.get(method))
    ax.set_xlabel("Calls to threshold (χ² < ε)")
    ax.set_ylabel("Count")
    ax.set_title("Sample efficiency: calls to convergence")
    ax.legend()

    ax = axes[1]
    for method in methods:
        subset = df[df["method"] == method]
        ax.scatter(subset["n_calls"], subset["chi2_min"],
                   alpha=0.5, label=method, color=COLORS.get(method))
    ax.set_xlabel("Total simulator calls")
    ax.set_ylabel("Best χ²")
    ax.set_title("Best calibration vs. budget")
    ax.legend()

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Figure saved to {args.out}")


if __name__ == "__main__":
    main()
