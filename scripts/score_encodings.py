#!/usr/bin/env python3
"""Scores one run of scripts/bench_encodings.sh and appends a row to encodings_runs.csv.

Arguments: the CSV, the encoding name, the index of the initial weights (1 to 3), the test file
("FEN label" per line), the predictions (one label per line, same order), the file written by
`/usr/bin/time -f "%e %M"` (wall seconds, peak memory in KB) and the training log (one line per
epoch). Columns: accuracy, macro F1 and recall of the five classes on the test set.
"""
import csv
import sys
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
COLUMNS = (["encoding", "width", "init_seed", "epochs", "accuracy", "macro_f1"]
           + ["recall_" + label.lower().replace(" ", "_") for label in LABELS]
           + ["train_seconds", "peak_rss_kb"])


def read_expected(path):
    with open(path) as f:
        return [" ".join(line.split()[6:]) for line in f if line.strip()]


def main(csv_path, encoding, init_seed, test, predictions, timing, log):
    expected = read_expected(test)
    with open(predictions) as f:
        predicted = [line.strip() for line in f if line.strip()]
    assert len(expected) == len(predicted), "one prediction per test position"
    seconds, peak = open(timing).read().split()
    epochs = sum(1 for line in open(log) if line.startswith("epoch "))

    recalls, f1s = [], []
    for label in LABELS:
        tp = sum(1 for t, p in zip(expected, predicted) if t == label and p == label)
        predicted_count = sum(1 for p in predicted if p == label)
        support = sum(1 for t in expected if t == label)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    accuracy = sum(t == p for t, p in zip(expected, predicted)) / len(expected)

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([encoding, encoding.rsplit("-", 1)[1], init_seed, epochs, f"{accuracy:.4f}",
                    f"{sum(f1s) / len(f1s):.4f}", *(f"{r:.4f}" for r in recalls), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:8])
