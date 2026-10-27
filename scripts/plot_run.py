#!/usr/bin/env python3
"""Draws the graphs of one run of scripts/train_final.sh. The metrics themselves come from
scripts/score.cpp (my_torch's report, ADR-0026); this script only draws and parses the training log.

Arguments: the training log of my_torch_analyzer and the run directory, which already holds the
confusion.csv written by scripts/score.cpp. Writes curve.csv (one row per epoch, from the log),
curve.png and confusion.png.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(log_file, directory):
    out = Path(directory)
    with open(out / "confusion.csv", newline="") as f:
        rows = list(csv.reader(f))
    labels = rows[0][1:]
    matrix = [[int(v) for v in row[1:]] for row in rows[1:]]
    total = sum(map(sum, matrix))
    accuracy = sum(matrix[i][i] for i in range(len(labels))) / total
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(matrix, cmap="Blues")
    peak = max(map(max, matrix))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > peak / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    ax.set_title(f"{total:,} positions, accuracy {accuracy:.1%}")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)

    pattern = re.compile(r"epoch (\d+)/\d+\s+loss ([\d.]+)\s+accuracy ([\d.]+)"
                         r"(?:\s+validation loss ([\d.]+)\s+validation accuracy ([\d.]+))?")
    curve = [m.groups() for m in map(pattern.search, open(log_file)) if m]
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss",
                    "validation_accuracy"])
        w.writerows(curve)
    if curve:
        epochs = [int(c[0]) for c in curve]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, [float(c[1]) for c in curve], label="training loss")
        if curve[0][3] is not None:
            ax.plot(epochs, [float(c[3]) for c in curve], label="validation loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:3])
