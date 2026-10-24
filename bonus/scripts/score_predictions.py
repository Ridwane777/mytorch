#!/usr/bin/env python3
"""Scores the predictions of the stalemate experiment.

Arguments: the tagged test file ("FEN label origin" per line), the predictions (one label per
line, same order) and the output directory. Writes stalemate_scores.csv (precision, recall and
support per class, for all positions, the dataset ones and the generated ones) and
stalemate_confusion.png.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black",
          "Stalemate"]


def read_tagged(path):
    rows = []
    with open(path) as f:
        for line in f:
            words = line.split()
            origin = words[-1]
            label = " ".join(words[6:-1])
            rows.append((label, origin))
    return rows


def scores(pairs):
    out = {}
    for label in LABELS:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        predicted = sum(1 for _, p in pairs if p == label)
        support = sum(1 for t, _ in pairs if t == label)
        out[label] = (tp / predicted if predicted else 0.0, tp / support if support else 0.0,
                      support)
    return out


def main(tagged, predictions, directory):
    rows = read_tagged(tagged)
    with open(predictions) as f:
        predicted = [line.strip() for line in f]
    assert len(rows) == len(predicted), "one prediction per test position"
    d = Path(directory)
    with open(d / "stalemate_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "class", "precision", "recall", "support"])
        for subset in ("all", "dataset", "generated"):
            pairs = [(t, p) for (t, o), p in zip(rows, predicted) if subset in ("all", o)]
            accuracy = sum(t == p for t, p in pairs) / len(pairs)
            w.writerow([subset, "accuracy", f"{accuracy:.4f}", "", len(pairs)])
            for label, (precision, recall, support) in scores(pairs).items():
                w.writerow([subset, label, f"{precision:.4f}", f"{recall:.4f}", support])

    counts = Counter((t, p) for (t, _), p in zip(rows, predicted))
    matrix = [[counts[(t, p)] for p in LABELS] for t in LABELS]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    ax.set_title("Six-class network on the test set (5,000 positions)")
    fig.tight_layout()
    fig.savefig(d / "stalemate_confusion.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:4])
