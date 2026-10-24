#!/usr/bin/env python3
"""Draws the graphs of scripts/bench_overfitting.cpp: reads the CSVs of a directory, writes PNGs."""
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


def main(directory):
    d = Path(directory)
    curves = defaultdict(lambda: ([], [], []))
    for row in read(d / "overfitting_curves.csv"):
        epoch, train, validation = curves[row["setting"]]
        epoch.append(int(row["epoch"]))
        train.append(float(row["train_loss"]))
        validation.append(float(row["validation_loss"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, (name, (epoch, train, validation)) in zip(axes.flat, curves.items()):
        ax.plot(epoch, train, label="training")
        ax.plot(epoch, validation, label="validation")
        ax.set_title(name)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("cross-entropy (median of 5 seeds)")
    axes[0][0].legend()
    fig.suptitle("Training against validation loss: no remedy and best setting of each remedy")
    fig.tight_layout()
    fig.savefig(d / "overfitting_curves.png", dpi=120)

    grid = read(d / "overfitting_grid.csv")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    baseline = next(float(r["test_loss"]) for r in grid if r["family"] == "baseline")
    for ax, family, label in ((axes[0], "l1", "L1 coefficient"),
                              (axes[1], "l2", "L2 coefficient"),
                              (axes[2], "elastic_net", "L1 = L2 coefficient"),
                              (axes[3], "dropout", "dropout rate")):
        rows = [r for r in grid if r["family"] == family]
        x = [float(r["setting"].split("=")[1]) for r in rows]
        y = [float(r["test_loss"]) for r in rows]
        low = [float(r["test_loss"]) - float(r["test_loss_min"]) for r in rows]
        high = [float(r["test_loss_max"]) - float(r["test_loss"]) for r in rows]
        ax.errorbar(x, y, yerr=[low, high], marker="o", capsize=3, label=family)
        ax.axhline(baseline, color="grey", linestyle="--", label="no remedy")
        if family != "dropout":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("test cross-entropy (median, min-max over 5 seeds)")
    fig.suptitle("Test loss against regularization strength")
    fig.tight_layout()
    fig.savefig(d / "overfitting_strength.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
