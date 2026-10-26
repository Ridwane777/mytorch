#!/usr/bin/env python3
"""Draws the graph of scripts/bench_augmentation.sh: reads augmentation_runs.csv of a directory and
writes augmentation.png (accuracy and recall per class, base arm against augmented arm). Bars are
medians of the runs, dots the individual runs."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"base": "#8a8a8a", "augmented": "#5b8def"}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def values(rows, arm, column):
    return [100 * float(r[column]) for r in rows if r["arm"] == arm]


def main(directory):
    d = Path(directory)
    rows = read(d / "augmentation_runs.csv")
    arms = [a for a in ("base", "augmented") if any(r["arm"] == a for r in rows)]
    columns = ["accuracy"] + [c for c in rows[0] if c.startswith("recall_")]
    names = ["accuracy"] + [c[len("recall_"):].replace("_", " ") for c in columns[1:]]
    sizes = defaultdict(set)
    for r in rows:
        sizes[r["arm"]].add(r["train_positions"])
    runs = len(values(rows, arms[0], "accuracy"))

    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, arm in enumerate(arms):
        xs = [j + i * width for j in range(len(columns))]
        ax.bar(xs, [statistics.median(values(rows, arm, c)) for c in columns], width,
               color=COLORS[arm], alpha=0.85,
               label=f"{arm} ({', '.join(sorted(sizes[arm]))} training positions)")
        for x, c in zip(xs, columns):
            points = values(rows, arm, c)
            ax.plot([x] * len(points), points, "k.", markersize=4)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(columns))], names, rotation=15)
    ax.set(ylabel="on the test set (%)", ylim=(0, 100),
           title=f"Symmetry augmentation: accuracy and recall per class (median of {runs} runs)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "augmentation.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
