#!/usr/bin/env python3
"""Scores one run of scripts/bench_encodings.sh and appends a row to encodings_runs.csv.

Arguments: the CSV, the encoding name, the index of the initial weights (1 to 3), the report.json
of scripts/score.cpp for the test set (the metrics come from my_torch, ADR-0026), the file written
by `/usr/bin/time -f "%e %M"` (wall seconds, peak memory in KB) and the training log (one line per
epoch). Columns: accuracy, macro F1 and recall of the five classes on the test set.
"""
import csv
import json
import sys
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
COLUMNS = (["encoding", "width", "init_seed", "epochs", "accuracy", "macro_f1"]
           + ["recall_" + label.lower().replace(" ", "_") for label in LABELS]
           + ["train_seconds", "peak_rss_kb"])


def main(csv_path, encoding, init_seed, report_path, timing, log):
    report = json.load(open(report_path))
    by_name = {c["name"]: c for c in report["classes"]}
    seconds, peak = open(timing).read().split()
    epochs = sum(1 for line in open(log) if line.startswith("epoch "))
    recalls = [by_name[label]["recall"] for label in LABELS]
    accuracy = report["accuracy"]
    macro_f1 = report["macro"]["f1"]

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([encoding, encoding.rsplit("-", 1)[1], init_seed, epochs, f"{accuracy:.4f}",
                    f"{macro_f1:.4f}", *(f"{r:.4f}" for r in recalls), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:7])
