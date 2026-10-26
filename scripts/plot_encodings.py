#!/usr/bin/env python3
"""Draws the graphs of scripts/bench_encodings.sh: reads encodings_runs.csv of a directory and
writes encodings_scores.png (accuracy, macro F1, recall per class) and encodings_cost.png
(training time, peak memory). Bars are medians of the runs, whiskers the minimum and maximum,
dots the individual runs."""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def by_encoding(rows, field):
    out = defaultdict(list)
    for row in rows:
        out[row["encoding"]].append(float(row[field]))
    return out


def bars(ax, groups, ylabel, title, scale=1.0, percent=False):
    names = list(groups)
    medians = [statistics.median(groups[n]) * scale for n in names]
    lows = [m - min(groups[n]) * scale for m, n in zip(medians, names)]
    highs = [max(groups[n]) * scale - m for m, n in zip(medians, names)]
    ax.bar(names, medians, yerr=[lows, highs], capsize=5, color="#5b8def", alpha=0.8)
    for i, n in enumerate(names):
        ax.plot([i] * len(groups[n]), [v * scale for v in groups[n]], "k.", markersize=5)
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.3)
    if percent:
        ax.set_ylim(0, 100)


def main(directory):
    d = Path(directory)
    rows = read(d / "encodings_runs.csv")
    runs = len(next(iter(by_encoding(rows, "accuracy").values())))

    fig, (ax_acc, ax_f1, ax_recall) = plt.subplots(1, 3, figsize=(17, 4.8),
                                                   gridspec_kw={"width_ratios": [1, 1, 1.8]})
    bars(ax_acc, by_encoding(rows, "accuracy"), "accuracy on the test set (%)",
         f"Accuracy (median of {runs} runs)", 100, True)
    bars(ax_f1, by_encoding(rows, "macro_f1"), "macro F1 on the test set (%)",
         "Macro F1, 5 classes", 100, True)
    recall_columns = [c for c in rows[0] if c.startswith("recall_")]
    encodings = list(by_encoding(rows, "accuracy"))
    width = 0.8 / len(encodings)
    for i, encoding in enumerate(encodings):
        medians = [100 * statistics.median(by_encoding(rows, c)[encoding]) for c in recall_columns]
        ax_recall.bar([j + i * width for j in range(len(recall_columns))], medians, width,
                      label=encoding)
    ax_recall.set_xticks([j + 0.4 - width / 2 for j in range(len(recall_columns))],
                         [c[len("recall_"):].replace("_", " ") for c in recall_columns],
                         rotation=15)
    ax_recall.set(ylabel="recall on the test set (%)", ylim=(0, 100),
                  title="Recall per class (median)")
    ax_recall.legend(fontsize=8)
    ax_recall.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "encodings_scores.png", dpi=130)

    fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.5))
    bars(ax_time, by_encoding(rows, "train_seconds"), "training wall time (s)",
         "Training time (reading, encoding and training)")
    bars(ax_mem, by_encoding(rows, "peak_rss_kb"), "peak memory (MB)", "Peak memory of --train",
         1 / 1024)
    fig.tight_layout()
    fig.savefig(d / "encodings_cost.png", dpi=130)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
