#!/usr/bin/env python3
"""Draws the curves of scripts/bench_optimizers.cpp: reads the CSVs of a directory, writes the PNGs."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def curves(rows, key):
    out = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for field in ("learning_rate", "loss", "accuracy"):
            if field in row:
                out[row[key]][field].append(float(row[field]))
    return out


def main(directory):
    d = Path(directory)

    optimizers = curves(read(d / "optimizers.csv"), "optimizer")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, c in optimizers.items():
        label = f"{name} (lr={c['learning_rate'][0]:g})"
        ax_loss.plot(c["loss"], label=label)
        ax_acc.plot(c["accuracy"], label=label)
    ax_loss.set(xlabel="epoch", ylabel="squared-error loss (training)", yscale="log",
                title="Loss at each optimizer's best rate (median of 5 seeds)")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (training)", title="Accuracy")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)
    ax_acc.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "optimizers.png", dpi=130)

    # Rate sensitivity: median final loss for every rate tried (a diverged run counts as 1).
    grid = defaultdict(list)
    for row in read(d / "optimizer_grid.csv"):
        loss = float(row["median_final_loss"])
        grid[row["optimizer"]].append((float(row["learning_rate"]), min(loss, 1.0) if loss == loss else 1.0))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, points in grid.items():
        ax.plot(*zip(*points), marker="o", label=name)
    ax.set(xlabel="learning rate", ylabel="median final loss (capped at 1)", xscale="log",
           yscale="log", title="Sensitivity to the learning rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "optimizer_rates.png", dpi=130)

    schedulers = curves(read(d / "schedulers.csv"), "scheduler")
    fig, (ax_rate, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, c in schedulers.items():
        ax_rate.plot(c["learning_rate"], label=name)
        ax_loss.plot(c["loss"], label=name)
    ax_rate.set(xlabel="epoch", ylabel="learning rate", title="Scheduled rate")
    ax_loss.set(xlabel="epoch", ylabel="squared-error loss (training)", yscale="log",
                title="Adam loss by scheduler (median of 5 seeds)")
    ax_rate.legend(fontsize=8)
    ax_rate.grid(alpha=0.3)
    ax_loss.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "schedulers.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
