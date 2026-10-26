#!/usr/bin/env python3
"""Scores one run of scripts/bench_augmentation.sh and appends a row to augmentation_runs.csv.

Arguments: the CSV, the arm ("base" or "augmented"), the number of training positions, the index of
the initial weights (1 to RUNS), the test file ("FEN label" per line), the predictions (one label
per line, same order), the file written by `/usr/bin/time -f "%e %M"` (wall seconds, peak memory
in KB). Columns: accuracy and recall of the five classes on the test set.
"""
import csv
import sys
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
COLUMNS = (["arm", "train_positions", "init_seed", "accuracy"]
           + ["recall_" + label.lower().replace(" ", "_") for label in LABELS]
           + ["train_seconds", "peak_rss_kb"])


def read_expected(path):
    with open(path) as f:
        return [" ".join(line.split()[6:]) for line in f if line.strip()]


def main(csv_path, arm, train_positions, init_seed, test, predictions, timing):
    expected = read_expected(test)
    with open(predictions) as f:
        predicted = [line.strip() for line in f if line.strip()]
    assert len(expected) == len(predicted), "one prediction per test position"
    seconds, peak = open(timing).read().split()

    recalls = []
    for label in LABELS:
        tp = sum(1 for t, p in zip(expected, predicted) if t == label and p == label)
        support = sum(1 for t in expected if t == label)
        recalls.append(tp / support if support else 0.0)
    accuracy = sum(t == p for t, p in zip(expected, predicted)) / len(expected)

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([arm, train_positions, init_seed, f"{accuracy:.4f}",
                    *(f"{r:.4f}" for r in recalls), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:8])
