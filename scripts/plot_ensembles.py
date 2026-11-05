#!/usr/bin/env python3
"""Table and graph of the ensembles benchmark (#40, docs/performance/27-ensembles.md), from the
directory written by scripts/bench_ensembles.sh: weights.csv (validation accuracy per CNN weight)
and test/NAME/{scores.txt,time.txt,bytes.txt} per model; the predictions of each model on the test
part (PREDICTIONS/NAME/predictions.txt) and the expected labels (TEST, the split's test.txt).
Writes DIR/ensembles.csv, DIR/mcnemar.csv (exact McNemar test of each ensemble against the CNN
alone, on the same positions) and DIR/ensembles.png.
Usage: plot_ensembles.py DIR CHOSEN_CNN_WEIGHT PREDICTIONS TEST"""
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


def models(weight):
    return [("dense", "dense alone"), ("cnn", "CNN alone"), ("average_1", "mean 1:1"),
            ("weighted", f"mean 1:{weight} (chosen on validation)"), ("vote", "vote 1:1"),
            ("stack", "stack (combiner trained on validation)")]


def scores(path):
    text = path.read_text()
    out = {"accuracy": float(re.search(r"accuracy: ([0-9.]+)", text).group(1)),
           "macro_f1": float(re.search(r"macro F1: ([0-9.]+)", text).group(1))}
    for c in CLASSES:
        out[c] = float(re.search(rf"^{c}: .*F1 ([0-9.]+)", text, re.M).group(1))
    return out


def mcnemar(directory, predictions, test):
    """Positions each ensemble fixes and breaks against the CNN alone, and the exact two-sided p."""
    truth = [" ".join(line.split()[6:]) for line in open(test)]
    read = lambda name: (Path(predictions) / name / "predictions.txt").read_text().splitlines()
    cnn = read("cnn")
    with open(directory / "mcnemar.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "fixed", "broken", "p_value"])
        for name in ("average_1", "weighted", "vote", "stack"):
            ours = read(name)
            fixed = sum(c != t and o == t for c, o, t in zip(cnn, ours, truth))
            broken = sum(c == t and o != t for c, o, t in zip(cnn, ours, truth))
            n = fixed + broken
            tail = sum(math.comb(n, i) for i in range(min(fixed, broken) + 1))
            p = min(1.0, 2 * tail / 2**n) if n else 1.0
            writer.writerow([name, fixed, broken, f"{p:.2g}"])
            print(f"{name} against the CNN: {fixed} fixed, {broken} broken, p = {p:.2g}")


def main(directory, weight, predictions, test):
    directory = Path(directory)
    mcnemar(directory, predictions, test)
    table = []
    for key, label in models(weight):
        d = directory / "test" / key
        seconds, kb = (d / "time.txt").read_text().split()
        table.append({"model": label, **scores(d / "scores.txt"), "predict_seconds": float(seconds),
                      "peak_mb": round(int(kb) / 1024),
                      "file_bytes": int((d / "bytes.txt").read_text())})
    fields = ["model", "accuracy", "macro_f1", *CLASSES, "predict_seconds", "peak_mb", "file_bytes"]
    with open(directory / "ensembles.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#a1d99b", "#31a354"]
    names = [r["model"] for r in table]

    ax = axes[0]
    bars = ax.barh(names, [r["accuracy"] for r in table], color=colors)
    for bar, r in zip(bars, table):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['accuracy']:.4f} (F1 {r['macro_f1']:.4f})", va="center", fontsize=8)
    low = min(r["accuracy"] for r in table)
    ax.set_xlim(low - 0.01, 1.0)
    ax.invert_yaxis()
    ax.set_title("Test accuracy (macro F1), 71,555 positions", fontsize=10)

    ax = axes[1]
    rows = list(csv.DictReader(open(directory / "weights.csv", newline="")))
    ax.plot([float(r["cnn_weight"]) for r in rows], [float(r["validation_accuracy"]) for r in rows],
            marker="o", color="#e6550d")
    ax.axvline(float(weight), color="#636363", linestyle="--", linewidth=1)
    ax.set_xlabel("weight of the CNN (dense = 1)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Choice of the weights, on the validation part only", fontsize=10)

    ax = axes[2]
    bars = ax.barh(names, [r["predict_seconds"] for r in table], color=colors)
    for bar, r in zip(bars, table):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['predict_seconds']:.1f} s, {r['file_bytes'] / 1e6:.1f} MB", va="center",
                fontsize=8)
    ax.set_xlim(0, max(r["predict_seconds"] for r in table) * 1.45)
    ax.invert_yaxis()
    ax.set_title("Prediction time (analyzer) and file size", fontsize=10)

    fig.suptitle("Ensembles of the dense network and the CNN against each alone")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(directory / "ensembles.png", dpi=100)
    for r in table:
        print(f"{r['model']}: accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f}, "
              f"{r['predict_seconds']} s, {r['file_bytes']:,} B")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:])
