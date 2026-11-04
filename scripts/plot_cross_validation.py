#!/usr/bin/env python3
"""Summarises and draws the cross-validation of scripts/bench_cross_validation.sh
(docs/performance/21-cross-validation.md, ADR-0039). The scores are computed by my_torch (ADR-0026);
this script only reads, summarises and draws.

Argument: the directory holding folds.csv (model, fold, train_rows, validation_rows, epochs, seconds,
accuracy, macro_precision, macro_recall, macro_f1; one line per model and fold). Writes, in the same
directory, summary.csv (per model and metric: number of folds, mean, sample standard deviation
(n - 1, as my_torch::summarize), and the 95 % confidence interval of the mean from a Student t
distribution with n - 1 degrees of freedom) and cross_validation.png (the folds, their mean and the
interval, per metric and model).
"""
import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
           ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]
COLORS = {"dense": "#e6550d", "cnn": "#3182bd"}
NAMES = {"dense": "Dense 901-256-128-5", "cnn": "CNN (2 convolutions)"}

# Two-sided 95 % critical values of the Student t distribution (t at 0.975), by degrees of freedom:
# the table avoids a dependency on scipy for a handful of folds.
T_975 = [12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
         2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
         2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423]


def t_critical(degrees):
    if degrees < 1:
        raise ValueError("a confidence interval needs at least two folds")
    return T_975[degrees - 1] if degrees <= len(T_975) else 1.96


def summarize(values):
    """Mean, sample standard deviation and 95 % interval of the mean."""
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    half = t_critical(n - 1) * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, mean - half, mean + half


def read(path):
    """folds.csv as {model: {column: [value per fold, in fold order]}}, models in file order."""
    models = OrderedDict()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            columns = models.setdefault(r["model"], {})
            for key, value in r.items():
                if key != "model":
                    columns.setdefault(key, []).append(float(value))
    return models


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    directory = Path(sys.argv[1])
    models = read(directory / "folds.csv")

    with open(directory / "summary.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["model", "metric", "folds", "mean", "std", "ci95_low", "ci95_high"])
        summaries = {}
        for model, columns in models.items():
            for metric, _ in METRICS + [("seconds", "")]:
                mean, std, low, high = summarize(columns[metric])
                summaries[model, metric] = (mean, std, low, high)
                out.writerow([model, metric, len(columns[metric])] +
                             [f"{v:.6f}" for v in (mean, std, low, high)])

    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6))
    for ax, (metric, title) in zip(axes, METRICS + [("seconds", "Training time per fold (s)")]):
        for x, (model, columns) in enumerate(models.items()):
            color = COLORS.get(model, "#555555")
            mean, std, low, high = summaries[model, metric]
            ax.scatter([x + 0.12 * ((i % 3) - 1) for i in range(len(columns[metric]))],
                       columns[metric], s=22, color=color, alpha=0.55, zorder=3,
                       label="one fold" if x == 0 else None)
            ax.errorbar([x], [mean], yerr=[[mean - low], [high - mean]], fmt="D", color="black",
                        markerfacecolor=color, markersize=7, capsize=7, zorder=4,
                        label="mean and 95 % interval" if x == 0 else None)
            ax.annotate(f"{mean:.4f}\n± {std:.4f}" if metric != "seconds" else f"{mean:.1f}\n± {std:.1f}",
                        (x, high), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([NAMES.get(m, m) for m in models], fontsize=7)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.25)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    any_model = next(iter(models.values()))
    k = len(any_model["accuracy"])
    fig.suptitle(f"Stratified {k}-fold cross-validation: mean ± sample standard deviation, interval "
                 "of the mean from a Student t distribution (95 %); training part of the split only",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "cross_validation.png", dpi=140, bbox_inches="tight")
    print(directory / "cross_validation.png")


if __name__ == "__main__":
    main()
