#!/usr/bin/env python3
"""Scores the predictions of the stalemate experiment with scripts/score.cpp, so the metrics come
from my_torch's report (ADR-0026); this script only cuts the subsets and draws.

Arguments: the score executable, the trained network, the tagged test file ("FEN label origin" per
line), the predictions (one label per line, same order) and the output directory. Writes
stalemate_scores.csv (precision, recall and support per class, for all positions, the dataset ones
and the generated ones) and stalemate_confusion.png.
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def score(executable, network, rows, predicted, directory):
    """Runs scripts/score.cpp on one subset and returns its report.json."""
    d = Path(directory)
    (d / "expected.txt").write_text("".join(f"{line}\n" for line, _ in rows))
    (d / "predicted.txt").write_text("".join(f"{p}\n" for p in predicted))
    subprocess.run([executable, network, d / "expected.txt", d / "predicted.txt", d], check=True)
    return json.load(open(d / "report.json"))


def main(executable, network, tagged, predictions, directory):
    rows = []
    for line in open(tagged):
        words = line.split()
        rows.append((" ".join(words[:-1]), words[-1]))  # "FEN label", origin
    with open(predictions) as f:
        predicted = [line.strip() for line in f]
    assert len(rows) == len(predicted), "one prediction per test position"
    d = Path(directory)
    with open(d / "stalemate_scores.csv", "w", newline="") as f, \
            tempfile.TemporaryDirectory() as work:
        w = csv.writer(f)
        w.writerow(["subset", "class", "precision", "recall", "support"])
        for subset in ("all", "dataset", "generated"):
            kept = [i for i, (_, origin) in enumerate(rows) if subset in ("all", origin)]
            report = score(executable, network, [rows[i] for i in kept],
                           [predicted[i] for i in kept], work)
            w.writerow([subset, "accuracy", f"{report['accuracy']:.4f}", "", len(kept)])
            for c in report["classes"]:
                w.writerow([subset, c["name"], f"{c['precision']:.4f}", f"{c['recall']:.4f}",
                            c["support"]])
            if subset == "all":
                labels = [c["name"] for c in report["classes"]]
                matrix = report["confusion"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center",
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("expected")
    ax.set_title("Six-class network on the test set (5,000 positions)")
    fig.tight_layout()
    fig.savefig(d / "stalemate_confusion.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:6])
