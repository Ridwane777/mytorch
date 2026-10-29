#!/usr/bin/env python3
"""Draws the graphs of issue #18 (docs/performance/16-benchmarks.md) from the CSV files written by
scripts/bench_report.sh, and the learning curves of the networks trained in full, already in
docs/performance/data.

Argument: the output directory (docs/performance/data/benchmarks). Writes learning_curves.png,
category_accuracy.png and timing.png.
"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("docs/performance/data")
RUNS = [
    ("final network, augmented (pushed)", DATA / "final-network/augmented"),
    ("final network, not augmented", DATA / "final-network/not-augmented"),
    ("search candidate (#10)", DATA / "candidate-network"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def learning_curves(out):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, run in RUNS:
        rows = read_csv(run / "curve.csv")
        epochs = [int(r["epoch"]) for r in rows]
        loss_ax.plot(epochs, [float(r["validation_loss"]) for r in rows], marker="o", label=name)
        acc_ax.plot(epochs, [float(r["validation_accuracy"]) for r in rows], marker="o",
                    label=name)
    loss_ax.set(xlabel="epoch", ylabel="validation loss", title="Validation loss, full data")
    acc_ax.set(xlabel="epoch", ylabel="validation accuracy", title="Validation accuracy, full data")
    for ax in (loss_ax, acc_ax):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=110)
    plt.close(fig)


def category_accuracy(out):
    groups = defaultdict(list)
    for r in read_csv(out / "category_accuracy.csv"):
        groups[r["group"]].append(r)
    order = ["checking piece", "pieces on the board", "side in check"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [7, 6, 3]})
    for ax, group in zip(axes, order):
        rows = groups[group]
        if group == "checking piece":
            rank = ["none", "pawn", "knight", "bishop", "rook", "queen", "double check"]
            rows.sort(key=lambda r: rank.index(r["category"]))
        names = [r["category"] for r in rows]
        values = [float(r["accuracy"]) for r in rows]
        bars = ax.bar(names, values, color="#4c78a8")
        for bar, r in zip(bars, rows):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{float(r['accuracy']):.3f}\n{int(r['positions']):,}", ha="center",
                    va="bottom", fontsize=8)
        ax.set_ylim(0.8, 1.02)
        ax.set_title(group)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy on the test split (label above: accuracy, positions)")
    fig.suptitle("Pushed network, test split, by category of position (categories from the oracle)")
    fig.tight_layout()
    fig.savefig(out / "category_accuracy.png", dpi=110)
    plt.close(fig)


def seconds_of(run_file, key):
    text = Path(run_file).read_text()
    return float(re.search(rf"{key}: ([0-9.]+) s", text).group(1))


def timing(out):
    fig, (train_ax, predict_ax, over_ax) = plt.subplots(1, 3, figsize=(15, 4.5))

    names = [name.replace(", ", "\n") for name, _ in RUNS]
    hours = [seconds_of(run / "run.txt", "training") / 3600 for _, run in RUNS]
    bars = train_ax.bar(names, hours, color="#72b7b2")
    for bar, h in zip(bars, hours):
        train_ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f} h", ha="center",
                      va="bottom")
    train_ax.set(ylabel="hours", title="Full training (one run each)")
    train_ax.tick_params(axis="x", labelsize=8)

    by_size = defaultdict(list)
    for r in read_csv(out / "predict_times.csv"):
        by_size[int(r["positions"])].append(float(r["seconds"]))
    sizes = sorted(by_size)
    rates = [[n / s for s in by_size[n]] for n in sizes]
    means = [statistics.mean(r) for r in rates]
    spreads = [statistics.stdev(r) if len(r) > 1 else 0 for r in rates]
    labels = [f"{n:,}" for n in sizes]
    predict_ax.bar(labels, means, yerr=spreads, capsize=4, color="#f58518")
    for i, n in enumerate(sizes):
        predict_ax.text(i, means[i], f"{means[i]:,.0f}/s\n{statistics.mean(by_size[n]):.2f} s",
                        ha="center", va="bottom", fontsize=8)
    predict_ax.set(xlabel="positions in the file", ylabel="positions per second",
                   title="--predict, whole command (mean ± sd)")

    by_variant = defaultdict(list)
    for r in read_csv(out / "overheads.csv"):
        by_variant[r["variant"]].append(float(r["seconds"]))
    variants = list(by_variant)
    means = [statistics.mean(by_variant[v]) for v in variants]
    spreads = [statistics.stdev(by_variant[v]) if len(by_variant[v]) > 1 else 0 for v in variants]
    over_ax.bar([v.replace(" and ", "\nand ") for v in variants], means, yerr=spreads, capsize=4,
                color="#54a24b")
    for i, m in enumerate(means):
        over_ax.text(i, m, f"{m:.2f} s\n{(m / means[0] - 1) * 100:+.1f} %", ha="center",
                     va="bottom", fontsize=8)
    over_ax.set(ylabel="seconds for 3 epochs", title="fit, final architecture, 8,192 rows")
    over_ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out / "timing.png", dpi=110)
    plt.close(fig)


def main(directory):
    out = Path(directory)
    learning_curves(out)
    category_accuracy(out)
    timing(out)


if __name__ == "__main__":
    main(sys.argv[1])
