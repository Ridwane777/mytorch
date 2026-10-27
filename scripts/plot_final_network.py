#!/usr/bin/env python3
"""Draws the final network graphs (docs/performance/10-final-network.md) from the run files that
scripts/train_final.sh writes, copied to docs/performance/data/final-network/<run>/: validation
loss per epoch and per-class F1 on the test split, with and without augmentation, and the
confusion matrix of the pushed network."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {"augmented": "with augmentation (pushed)", "not-augmented": "without augmentation"}


def curve(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def class_f1(path):
    scores = {}
    for line in open(path):
        if ": precision" in line:
            name, rest = line.split(": precision")
            scores[name] = float(rest.split("F1")[1].split(",")[0])
    return scores


def main(directory):
    d = Path(directory)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    for run, label in RUNS.items():
        rows = curve(d / run / "curve.csv")
        epochs = [int(r["epoch"]) for r in rows]
        loss = [float(r["validation_loss"]) for r in rows]
        line, = ax1.plot(epochs, loss, marker="o", label=label)
        best = loss.index(min(loss))
        ax1.scatter([epochs[best]], [loss[best]], s=150, facecolors="none",
                    edgecolors=line.get_color(), linewidths=2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation loss (cross-entropy)")
    ax1.set_title("Validation loss; circled: best epoch, restored by early stopping")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    scores = {run: class_f1(d / run / "scores.txt") for run in RUNS}
    names = list(scores["augmented"])
    width = 0.38
    for i, (run, label) in enumerate(RUNS.items()):
        xs = [k + (i - 0.5) * width for k in range(len(names))]
        values = [scores[run][n] for n in names]
        ax2.bar(xs, values, width, label=label)
        for x, v in zip(xs, values):
            ax2.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(names)), names, rotation=20)
    ax2.set_ylim(0.8, 1.0)
    ax2.set_ylabel("F1 on the test split")
    ax2.set_title("Per-class F1, 71,555 test positions")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "final_network.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data/final-network")
