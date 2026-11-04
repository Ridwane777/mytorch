#!/usr/bin/env python3
"""Draws docs/performance/data/cnn_vs_dense.png (docs/performance/20-cnn.md) from the CSV written by
scripts/bench_cnn.sh: test accuracy and macro F1, per-class F1, training time, prediction time,
parameters and file size, for the CNN and the dense network.
Usage: plot_cnn.py cnn_vs_dense.csv output.png"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = [("nothing", "Nothing"), ("check_white", "Check White"), ("check_black", "Check Black"),
           ("checkmate_white", "Checkmate White"), ("checkmate_black", "Checkmate Black")]
COLORS = {"cnn": "#3182bd", "dense": "#e6550d"}
LABELS = {"cnn": "CNN (10 % of the training data)",
          "dense": "Dense 901-256-128-5 (10 % of the training data)"}


def read(path):
    with open(path, newline="") as f:
        return {r["metric"]: {"cnn": float(r["cnn"]), "dense": float(r["dense"])}
                for r in csv.DictReader(f)}


def grouped(ax, groups, data, title, low=None):
    """Bars of both networks for each group (a metric name and its label)."""
    width = 0.38
    for k, net in enumerate(("cnn", "dense")):
        xs = [i + (k - 0.5) * width for i in range(len(groups))]
        values = [data[metric][net] for metric, _ in groups]
        bars = ax.bar(xs, values, width, color=COLORS[net], label=LABELS[net])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7, rotation=90)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([label for _, label in groups], fontsize=8,
                       rotation=15 if len(groups) > 2 else 0, ha="right" if len(groups) > 2 else "center")
    ax.set_title(title, fontsize=10)
    if low is not None:
        ax.set_ylim(low, 1.0 + (1.0 - low) * 0.18)
    ax.grid(axis="y", alpha=0.3)


def single(ax, data, metric, title, scale=1.0, unit="", fmt="{:,.0f}"):
    for k, net in enumerate(("cnn", "dense")):
        v = data[metric][net] * scale
        ax.bar(k, v, 0.6, color=COLORS[net])
        ax.text(k, v, fmt.format(v) + unit, ha="center", va="bottom", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CNN", "Dense"], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, max(data[metric].values()) * scale * 1.2)
    ax.grid(axis="y", alpha=0.3)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data = read(sys.argv[1])
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.35)
    grouped(fig.add_subplot(grid[0, 0:2]), [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")], data,
            "Test accuracy and macro F1", low=0.7)
    ax = fig.add_subplot(grid[0, 2:4])
    grouped(ax, [(f"f1_{key}", label) for key, label in CLASSES], data, "Test F1 per class",
            low=0.7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.95))
    single(fig.add_subplot(grid[1, 0]), data, "train_seconds", "Training time (minutes)", 1 / 60,
           "", "{:,.1f}")
    single(fig.add_subplot(grid[1, 1]), data, "predict_seconds_median",
           "Prediction of the test split (s)", 1, "", "{:.2f}")
    single(fig.add_subplot(grid[1, 2]), data, "parameters", "Parameters", 1, "", "{:,.0f}")
    single(fig.add_subplot(grid[1, 3]), data, "file_bytes", "Network file (MB)", 1e-6, "", "{:.2f}")
    fig.suptitle("Convolutional network against the dense network, both trained on 10 % of the "
                 f"training data, test split ({int(data['test_positions']['cnn']):,} positions)",
                 fontsize=12)
    fig.savefig(sys.argv[2], dpi=140, bbox_inches="tight")
    print(sys.argv[2])


if __name__ == "__main__":
    main()
