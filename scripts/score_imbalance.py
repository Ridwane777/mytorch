#!/usr/bin/env python3
"""Scores one run of scripts/bench_imbalance.sh and appends a row to imbalance_runs.csv.

Arguments: the CSV, the option name (none, weighted, oversample, undersample), the index of the
initial weights (1 to 3), the number of positions of the training file of the option, the test file
("FEN label" per line), the predictions (one label per line, same order) and the file written by
`/usr/bin/time -f "%e %M"` (wall seconds, peak memory in KB).
Columns: accuracy, macro F1, F1 of each of the five classes and F1 of the three families
(Nothing, Check = both colours, Checkmate = both colours) on the test set.
"""
import csv
import sys
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
FAMILIES = ["Nothing", "Check", "Checkmate"]


def column(name):
    return name.lower().replace(" ", "_")


COLUMNS = (["option", "init_seed", "train_size", "accuracy", "macro_f1"]
           + ["f1_" + column(label) for label in LABELS]
           + ["f1_family_" + column(family) for family in FAMILIES]
           + ["train_seconds", "peak_rss_kb"])


def read_expected(path):
    with open(path) as f:
        return [" ".join(line.split()[6:]) for line in f if line.strip()]


def family(label):
    return label.split()[0]


def f1(expected, predicted, wanted):
    """F1 of the class `wanted`, 0 when it is never predicted or never present."""
    tp = sum(1 for t, p in zip(expected, predicted) if t == wanted and p == wanted)
    predicted_count = sum(1 for p in predicted if p == wanted)
    support = sum(1 for t in expected if t == wanted)
    precision = tp / predicted_count if predicted_count else 0.0
    recall = tp / support if support else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main(csv_path, option, init_seed, train_size, test, predictions, timing):
    expected = read_expected(test)
    with open(predictions) as f:
        predicted = [line.strip() for line in f if line.strip()]
    assert len(expected) == len(predicted), "one prediction per test position"
    seconds, peak = open(timing).read().split()

    f1s = [f1(expected, predicted, label) for label in LABELS]
    family_expected = [family(t) for t in expected]
    family_predicted = [family(p) for p in predicted]
    family_f1s = [f1(family_expected, family_predicted, name) for name in FAMILIES]
    accuracy = sum(t == p for t, p in zip(expected, predicted)) / len(expected)

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([option, init_seed, train_size, f"{accuracy:.4f}", f"{sum(f1s) / len(f1s):.4f}",
                    *(f"{v:.4f}" for v in f1s), *(f"{v:.4f}" for v in family_f1s), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:8])
