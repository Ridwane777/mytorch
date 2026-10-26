#!/usr/bin/env python3
"""Scores the predictions of scripts/train_final.sh on the test split.

Arguments: the test file ("FEN label" per line), the predictions (one label per line, same order),
the training log of my_torch_analyzer and the output directory. Writes scores.txt (accuracy, macro
F1, precision, recall and F1 per class), confusion.csv, curve.csv, confusion.png and curve.png.
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(test_file, predictions_file, log_file, directory):
    out = Path(directory)
    expected = [" ".join(line.split()[6:]) for line in open(test_file) if line.strip()]
    predicted = [line.strip() for line in open(predictions_file) if line.strip()]
    assert len(expected) == len(predicted), "one prediction per test position"
    labels = sorted(set(expected) | set(predicted),
                    key=lambda l: ["Nothing", "Check White", "Check Black", "Checkmate White",
                                   "Checkmate Black"].index(l) if l in (
                        "Nothing", "Check White", "Check Black", "Checkmate White",
                        "Checkmate Black") else 99)
    pairs = Counter(zip(expected, predicted))

    lines = [f"test positions: {len(expected)}"]
    accuracy = sum(pairs[(l, l)] for l in labels) / len(expected)
    f1s = []
    rows = []
    for label in labels:
        tp = pairs[(label, label)]
        support = sum(pairs[(label, p)] for p in labels)
        guessed = sum(pairs[(t, label)] for t in labels)
        precision = tp / guessed if guessed else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        rows.append(f"{label}: precision {precision:.4f}, recall {recall:.4f}, F1 {f1:.4f}, "
                    f"support {support}")
    lines.append(f"accuracy: {accuracy:.4f}")
    lines.append(f"macro F1: {sum(f1s) / len(f1s):.4f}")
    (out / "scores.txt").write_text("\n".join(lines + rows) + "\n")

    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expected \\ predicted"] + labels)
        for t in labels:
            w.writerow([t] + [pairs[(t, p)] for p in labels])
    matrix = [[pairs[(t, p)] for p in labels] for t in labels]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(matrix, cmap="Blues")
    peak = max(map(max, matrix))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > peak / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    ax.set_title(f"Test split ({len(expected):,} positions), accuracy {accuracy:.1%}")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)

    pattern = re.compile(r"epoch (\d+)/\d+\s+loss ([\d.]+)\s+accuracy ([\d.]+)"
                         r"(?:\s+validation loss ([\d.]+)\s+validation accuracy ([\d.]+))?")
    curve = [m.groups() for m in map(pattern.search, open(log_file)) if m]
    with open(out / "curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_accuracy", "validation_loss",
                    "validation_accuracy"])
        w.writerows(curve)
    if curve:
        epochs = [int(c[0]) for c in curve]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, [float(c[1]) for c in curve], label="training loss")
        if curve[0][3] is not None:
            ax.plot(epochs, [float(c[3]) for c in curve], label="validation loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:5])
