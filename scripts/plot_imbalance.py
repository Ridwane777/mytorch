#!/usr/bin/env python3
"""Draws the graphs of scripts/bench_imbalance.sh: reads imbalance_runs.csv of a directory and
writes imbalance_f1.png (F1 of each of the five classes and macro F1 for each option) and
imbalance_family_f1.png (F1 of the three families Nothing, Check, Checkmate). Bars are the mean of
the runs (different initial weights), whiskers the minimum and maximum, dots the individual runs."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPTIONS = ["none", "weighted", "oversample", "undersample"]
COLORS = ["#8c8c8c", "#5b8def", "#e8833a", "#4aa564"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def values(rows, option, field):
    return [float(r[field]) for r in rows if r["option"] == option]


def grouped(ax, rows, fields, names, title):
    options = [o for o in OPTIONS if any(r["option"] == o for r in rows)]
    width = 0.8 / len(options)
    for i, option in enumerate(options):
        xs = [j + i * width for j in range(len(fields))]
        runs = [values(rows, option, field) for field in fields]
        means = [100 * statistics.mean(v) for v in runs]
        lows = [m - 100 * min(v) for m, v in zip(means, runs)]
        highs = [100 * max(v) - m for m, v in zip(means, runs)]
        ax.bar(xs, means, width, yerr=[lows, highs], capsize=3, label=option,
               color=COLORS[OPTIONS.index(option)], alpha=0.85)
        for x, v in zip(xs, runs):
            ax.plot([x] * len(v), [100 * y for y in v], "k.", markersize=4)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(fields))], names, rotation=12)
    ax.set(ylabel="F1 on the test set (%)", ylim=(0, 100), title=title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main(directory):
    d = Path(directory)
    rows = read(d / "imbalance_runs.csv")
    runs = len({r["init_seed"] for r in rows})

    f1_columns = [c for c in rows[0] if c.startswith("f1_") and not c.startswith("f1_family_")]
    names = [c[len("f1_"):].replace("_", " ") for c in f1_columns] + ["macro F1"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    grouped(ax, rows, f1_columns + ["macro_f1"], names,
            f"F1 per class by imbalance option (mean of {runs} runs, whiskers min to max)")
    fig.tight_layout()
    fig.savefig(d / "imbalance_f1.png", dpi=130)

    family_columns = [c for c in rows[0] if c.startswith("f1_family_")]
    names = [c[len("f1_family_"):] for c in family_columns]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    grouped(ax, rows, family_columns, names,
            f"F1 per family by imbalance option (mean of {runs} runs)")
    fig.tight_layout()
    fig.savefig(d / "imbalance_family_f1.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
