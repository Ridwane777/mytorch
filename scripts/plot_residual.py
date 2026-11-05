#!/usr/bin/env python3
"""Table and graph of the residual benchmark (#31, docs/performance/28-residual.md), from the
directory written by scripts/bench_residual.sh: NAME/{epochs.txt,train_time.txt,predict_time.txt,
scores.txt,bytes.txt} per network.
Writes DIR/residual.csv and DIR/residual.png.
Usage: plot_residual.py DIR NAME..."""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
LABELS = {"cnn": "CNN, 2 convolutions (reference)",
          "cnn_deep": "6 convolutions, no shortcut (control)",
          "cnn_residual": "6 convolutions, 2 residual blocks"}
COLORS = {"cnn": "#3182bd", "cnn_deep": "#9e9ac8", "cnn_residual": "#e6550d"}
EPOCH = re.compile(r"epoch (\d+)/\d+\s+loss ([0-9.]+)\s+accuracy ([0-9.]+)\s+validation loss "
                   r"([0-9.]+)\s+validation accuracy ([0-9.]+)")


def scores(path):
    text = path.read_text()
    out = {"accuracy": float(re.search(r"accuracy: ([0-9.]+)", text).group(1)),
           "macro_f1": float(re.search(r"macro F1: ([0-9.]+)", text).group(1))}
    for c in CLASSES:
        out[c] = float(re.search(rf"^{c}: .*F1 ([0-9.]+)", text, re.M).group(1))
    return out


def main(directory, names):
    directory = Path(directory)
    table, curves = [], {}
    for name in names:
        d = directory / name
        epochs = [tuple(float(g) for g in m.groups()) for m in
                  EPOCH.finditer((d / "epochs.txt").read_text())]
        curves[name] = epochs
        train_seconds, train_kb = (d / "train_time.txt").read_text().split()
        predict_seconds, _ = (d / "predict_time.txt").read_text().split()
        table.append({"model": LABELS.get(name, name), **scores(d / "scores.txt"),
                      "final_train_loss": epochs[-1][1], "final_validation_loss": epochs[-1][3],
                      "best_validation_accuracy": max(e[4] for e in epochs),
                      "train_seconds": float(train_seconds),
                      "seconds_per_epoch": round(float(train_seconds) / len(epochs), 1),
                      "train_peak_mb": round(int(train_kb) / 1024),
                      "predict_seconds": float(predict_seconds),
                      "file_bytes": int((d / "bytes.txt").read_text())})
    fields = list(table[0])
    with open(directory / "residual.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name in names:
        e = curves[name]
        x = [row[0] for row in e]
        axes[0].plot(x, [row[1] for row in e], color=COLORS.get(name), linestyle="--",
                     label=f"{LABELS.get(name, name)}: training")
        axes[0].plot(x, [row[3] for row in e], color=COLORS.get(name),
                     label=f"{LABELS.get(name, name)}: validation")
        axes[1].plot(x, [row[4] for row in e], color=COLORS.get(name), marker="o", markersize=3,
                     label=LABELS.get(name, name))
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss per epoch (dashed: training)", fontsize=10)
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("Validation accuracy per epoch", fontsize=10)
    axes[1].legend(fontsize=7)

    ax = axes[2]
    names_shown = [r["model"] for r in table]
    bars = ax.barh(names_shown, [r["accuracy"] for r in table],
                   color=[COLORS.get(n) for n in names])
    for bar, r in zip(bars, table):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['accuracy']:.4f}, {r['seconds_per_epoch']:.0f} s/epoch", va="center",
                fontsize=8)
    low = min(r["accuracy"] for r in table)
    ax.set_xlim(low - 0.02, 1.0)
    ax.invert_yaxis()
    ax.set_title("Test accuracy (71,555 positions) and time per epoch", fontsize=10)

    fig.suptitle("Residual blocks against the same depth without shortcuts, same budget")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "residual.png", dpi=100)
    for r in table:
        print(f"{r['model']}: test accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f}, "
              f"{r['seconds_per_epoch']} s/epoch")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
