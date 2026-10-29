#!/usr/bin/env python3
"""Draws the learning curves of the normalization benchmark (#30, scripts/bench_normalization.sh).

Argument: the output directory, holding one folder per run (shallow_none, shallow_batch, ...) with
its curve.csv, scores.txt and run.txt. Writes normalization.png and summary.csv.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEPTHS = [("shallow", "final architecture, 901 → 256 → 128 → 5"),
          ("deep", "deep network, 6 hidden layers of 256")]
NORMS = [("none", "no normalization"), ("batch", "BatchNorm"), ("layer", "LayerNorm")]


def curve(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def figure(path, pattern):
    match = re.search(pattern, Path(path).read_text())
    return float(match.group(1)) if match else float("nan")


def main(directory):
    out = Path(directory)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    summary = []
    for row, (depth, title) in enumerate(DEPTHS):
        for norm, label in NORMS:
            run = out / f"{depth}_{norm}"
            rows = curve(run / "curve.csv")
            epochs = [int(r["epoch"]) for r in rows]
            axes[row][0].plot(epochs, [float(r["train_loss"]) for r in rows], marker="o",
                              label=label)
            axes[row][1].plot(epochs, [float(r["validation_loss"]) for r in rows], marker="o",
                              label=label)
            axes[row][2].plot(epochs, [float(r["validation_accuracy"]) for r in rows],
                              marker="o", label=label)
            seconds = figure(run / "run.txt", r"training: ([0-9.]+) s")
            summary.append({
                "network": depth, "normalization": norm, "epochs": len(rows),
                "training_seconds": seconds,
                "seconds_per_epoch": round(seconds / len(rows), 1),
                "best_validation_loss": min(float(r["validation_loss"]) for r in rows),
                "validation_accuracy": figure(run / "scores.txt", r"accuracy: ([0-9.]+)"),
                "validation_macro_f1": figure(run / "scores.txt", r"macro F1: ([0-9.]+)"),
            })
        for ax, what in zip(axes[row], ["training loss", "validation loss",
                                        "validation accuracy"]):
            ax.set(xlabel="epoch", ylabel=what, title=f"{what}, {title}")
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("Normalization at 10 % of the data (validation split)")
    fig.tight_layout()
    fig.savefig(out / "normalization.png", dpi=110)
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main(sys.argv[1])
