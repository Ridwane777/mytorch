#!/usr/bin/env python3
"""Table and graphs of the attention benchmark (#32, docs/performance/29-attention.md), from the
directory written by scripts/bench_attention.sh: NAME/{epochs.txt,train_time.txt,predict_time.txt,
scores.txt,bytes.txt} per network, and mcnemar_NAME.txt (each network against the CNN).
Writes DIR/attention.csv and DIR/attention.png. Sizes are in MiB (1 MiB = 1,048,576 bytes).
Usage: plot_attention.py DIR NAME..."""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
LABELS = {"dense": "Dense 901-256-128-5",
          "cnn": "CNN, 2 convolutions",
          "transformer1": "Transformer, 1 block",
          "transformer2": "Transformer, 2 blocks"}
COLORS = {"dense": "#74c476", "cnn": "#3182bd", "transformer1": "#e6550d",
          "transformer2": "#a63603"}
EPOCH = re.compile(r"epoch (\d+)/\d+\s+loss ([0-9.]+)\s+accuracy ([0-9.]+)\s+validation loss "
                   r"([0-9.]+)\s+validation accuracy ([0-9.]+)")
MIB = 1024 * 1024


def scores(path):
    text = path.read_text()
    out = {"positions": int(re.search(r"positions scored: (\d+)", text).group(1)),
           "accuracy": float(re.search(r"accuracy: ([0-9.]+)", text).group(1)),
           "macro_f1": float(re.search(r"macro F1: ([0-9.]+)", text).group(1))}
    for c in CLASSES:
        out[f"F1 {c}"] = float(re.search(rf"^{c}: .*F1 ([0-9.]+)", text, re.M).group(1))
    return out


def mcnemar(path):
    if not path.exists():
        return {}
    values = dict(line.split(": ") for line in path.read_text().splitlines())
    return {"fixed_vs_cnn": int(values["fixed"]), "broken_vs_cnn": int(values["broken"]),
            "mcnemar_p_vs_cnn": values["p"]}


def main(directory, names):
    directory = Path(directory)
    table, curves = [], {}
    for name in names:
        d = directory / name
        epochs = [tuple(float(g) for g in m.groups()) for m in
                  EPOCH.finditer((d / "epochs.txt").read_text())]
        curves[name] = epochs
        train_seconds, train_kib = (d / "train_time.txt").read_text().split()
        predict_seconds, predict_kib = (d / "predict_time.txt").read_text().split()
        table.append({"network": name, "model": LABELS.get(name, name),
                      **scores(d / "scores.txt"),
                      "final_train_loss": epochs[-1][1], "final_validation_loss": epochs[-1][3],
                      "best_validation_accuracy": max(e[4] for e in epochs),
                      "train_seconds": float(train_seconds),
                      "seconds_per_epoch": round(float(train_seconds) / len(epochs), 1),
                      "train_peak_mib": round(int(train_kib) / 1024),
                      "predict_seconds": float(predict_seconds),
                      "predict_peak_mib": round(int(predict_kib) / 1024),
                      "file_mib": round(int((d / "bytes.txt").read_text()) / MIB, 2),
                      **mcnemar(directory / f"mcnemar_{name}.txt")})
    fields = []
    for row in table:
        fields += [k for k in row if k not in fields]
    with open(directory / "attention.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    positions = table[0]["positions"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    loss, accuracy, test, cost = axes.flat
    for name in names:
        e = curves[name]
        x = [row[0] for row in e]
        loss.plot(x, [row[1] for row in e], color=COLORS.get(name), linestyle="--",
                  label=f"{LABELS.get(name, name)}: training")
        loss.plot(x, [row[3] for row in e], color=COLORS.get(name),
                  label=f"{LABELS.get(name, name)}: validation")
        accuracy.plot(x, [row[4] for row in e], color=COLORS.get(name), marker="o", markersize=3,
                      label=LABELS.get(name, name))
    epochs_count = len(curves[names[0]])
    for ax in (loss, accuracy):
        ax.set_xlabel("epoch")
        ax.set_xticks(range(1, epochs_count + 1))
    loss.set_ylabel("cross-entropy loss (log scale)")
    loss.set_yscale("log")
    loss.set_title("Loss per epoch (dashed: training, solid: validation)", fontsize=10)
    loss.legend(fontsize=7)
    accuracy.set_ylabel("validation accuracy")
    accuracy.set_title("Validation accuracy per epoch", fontsize=10)
    accuracy.legend(fontsize=8, loc="lower right")

    def bars(ax, key, ylabel, title, fmt, low=None):
        values = [r[key] for r in table]
        shown = ax.bar(range(len(table)), values, color=[COLORS.get(n) for n in names],
                       label=None)
        for bar, value in zip(shown, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value),
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels([LABELS.get(n, n).replace(", ", ",\n") for n in names], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if low is not None:
            ax.set_ylim(low, 1.0)
        ax.legend(shown, [LABELS.get(n, n) for n in names], fontsize=7, loc="lower right")

    bars(test, "accuracy", "test accuracy", f"Test accuracy ({positions:,} positions, read once)",
         "{:.4f}", low=min(r["accuracy"] for r in table) - 0.03)
    bars(cost, "seconds_per_epoch", "training time per epoch (s)",
         "Training time per epoch (peak memory in the bar labels, MiB)", "{:.0f} s")
    for bar, r in zip(cost.patches, table):
        cost.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                  f"{r['train_peak_mib']:,} MiB", ha="center", va="center", fontsize=8,
                  color="white")

    fig.suptitle("A mini transformer against the CNN and the dense network, same budget")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(directory / "attention.png", dpi=100)
    for r in table:
        print(f"{r['model']}: test accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f}, "
              f"{r['seconds_per_epoch']} s/epoch, {r['train_peak_mib']} MiB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
