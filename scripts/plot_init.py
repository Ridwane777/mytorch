#!/usr/bin/env python3
"""Draws the curves of scripts/bench_init.cpp: reads init_curves.csv and init_summary.csv of a
directory, writes init_curves.png (learning curves) and init_final.png (final loss)."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = (("tanh", "tanh hidden layers"), ("relu", "ReLU hidden layers"))


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(directory):
    d = Path(directory)

    # family -> init -> column -> values over the epochs
    curves = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in read(d / "init_curves.csv"):
        c = curves[row["family"]][row["init"]]
        for field in ("median_loss", "min_loss", "max_loss"):
            c[field].append(float(row[field]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (family, title) in zip(axes, FAMILIES):
        for name, c in curves[family].items():
            line, = ax.plot(c["median_loss"], label=name, linewidth=2 if name == "uniform" else 1.2,
                            linestyle="--" if name == "uniform" else "-")
            ax.fill_between(range(len(c["median_loss"])), c["min_loss"], c["max_loss"],
                            color=line.get_color(), alpha=0.12)
        ax.set(xlabel="epoch", title=f"2-64-64-64-1, {title} (median and min-max of 5 seeds)",
               yscale="log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("squared-error loss (training)")
    fig.tight_layout()
    fig.savefig(d / "init_curves.png", dpi=130)

    # Final loss per initialisation: median, with the min-max range of the seeds.
    summary = defaultdict(list)
    for row in read(d / "init_summary.csv"):
        summary[row["family"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (family, title) in zip(axes, FAMILIES):
        rows = summary[family]
        names = [r["init"] for r in rows]
        med = [float(r["median_final_loss"]) for r in rows]
        low = [m - float(r["min_final_loss"]) for m, r in zip(med, rows)]
        high = [float(r["max_final_loss"]) - m for m, r in zip(med, rows)]
        colors = ["tab:gray" if n == "uniform" else "tab:blue" for n in names]
        ax.bar(names, med, yerr=[low, high], color=colors, capsize=4)
        ax.set(title=f"Final loss after 400 epochs, {title}", yscale="log")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("final squared-error loss (median, min-max)")
    fig.tight_layout()
    fig.savefig(d / "init_final.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
