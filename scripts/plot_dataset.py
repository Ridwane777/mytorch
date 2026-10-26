#!/usr/bin/env python3
"""Draws the dataset charts of scripts/dataset_split.py: reads the CSVs of a directory, writes the PNGs.

Usage: python3 scripts/plot_dataset.py [statistics directory (default docs/performance/data)]
Writes dataset_labels.png, dataset_pieces.png and dataset_split.png.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
CLASSES = ["Nothing", "Check", "Checkmate"]
PIECES = "PNBRQKpnbrqk"
SPLITS = ["train", "validation", "test"]


def read(path):
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except OSError as e:
        sys.exit(f"plot_dataset.py: {e} (run scripts/dataset_split.py first)")


def bar_labels(ax, bars):
    for bar in bars:
        ax.annotate(f"{int(bar.get_height()):,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=7)


def labels_chart(d):
    counts = {row["label"]: int(row["count"]) for row in read(d / "dataset_labels.csv")}
    side = defaultdict(lambda: defaultdict(int))
    for row in read(d / "dataset_pieces.csv"):
        if row["metric"] == "side":
            side[row["class"]][row["value"]] += int(row["count"])

    fig, (ax_labels, ax_side) = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = ax_labels.bar(range(len(LABELS)), [counts.get(label, 0) for label in LABELS])
    bar_labels(ax_labels, bars)
    ax_labels.set_xticks(range(len(LABELS)), LABELS, rotation=20, fontsize=8)
    ax_labels.set(ylabel="lines", title="Lines per label")
    ax_labels.grid(axis="y", alpha=0.3)

    width = 0.38
    for i, (letter, name) in enumerate([("w", "White to move"), ("b", "Black to move")]):
        ax_side.bar([x + (i - 0.5) * width for x in range(len(CLASSES))],
                    [side[c][letter] for c in CLASSES], width, label=name)
    ax_side.set_xticks(range(len(CLASSES)), CLASSES)
    ax_side.set(ylabel="lines", title="Side to move per class")
    ax_side.legend(fontsize=8)
    ax_side.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "dataset_labels.png", dpi=130)
    plt.close(fig)


def pieces_chart(d):
    hist = defaultdict(lambda: defaultdict(dict))  # class -> metric -> value -> count
    for row in read(d / "dataset_pieces.csv"):
        if row["metric"] != "side":
            hist[row["class"]][row["metric"]][int(row["value"])] = int(row["count"])

    fig, (ax_total, ax_mean) = plt.subplots(1, 2, figsize=(12, 4.5))
    for cls in CLASSES:
        h = hist[cls]["total"]
        n = sum(h.values())
        if n:
            xs = sorted(h)
            ax_total.plot(xs, [100 * h[x] / n for x in xs], marker="o", markersize=3, label=cls)
    ax_total.set(xlabel="pieces on the board", ylabel="share of the class's lines (%)",
                 title="Number of pieces per class")
    ax_total.legend(fontsize=8)
    ax_total.grid(alpha=0.3)

    width = 0.27
    for i, cls in enumerate(CLASSES):
        means = []
        for piece in PIECES:
            h = hist[cls][piece]
            n = sum(h.values())
            means.append(sum(v * c for v, c in h.items()) / n if n else 0.0)
        ax_mean.bar([x + (i - 1) * width for x in range(len(PIECES))], means, width, label=cls)
    ax_mean.set_xticks(range(len(PIECES)), list(PIECES))
    ax_mean.set(xlabel="piece (upper case = White)", ylabel="mean count per position",
                title="Pieces by type per class")
    ax_mean.legend(fontsize=8)
    ax_mean.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "dataset_pieces.png", dpi=130)
    plt.close(fig)


def split_chart(d):
    counts = defaultdict(dict)
    for row in read(d / "dataset_split.csv"):
        counts[row["split"]][row["label"]] = int(row["count"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.27
    for i, name in enumerate(SPLITS):
        bars = ax.bar([x + (i - 1) * width for x in range(len(LABELS))],
                      [counts[name].get(label, 0) for label in LABELS], width, label=name)
        bar_labels(ax, bars)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20, fontsize=8)
    ax.set(ylabel="positions", title="Positions per label in each split")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "dataset_split.png", dpi=130)
    plt.close(fig)


def main(directory):
    d = Path(directory)
    labels_chart(d)
    pieces_chart(d)
    split_chart(d)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
