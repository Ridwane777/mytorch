#!/usr/bin/env python3
"""Scores one run of scripts/bench_augmentation.sh and appends a row to augmentation_runs.csv.

Arguments: the CSV, the arm ("base" or "augmented"), the number of training positions, the index of
the initial weights (1 to RUNS), the report.json of scripts/score.cpp for the test set (the metrics
come from my_torch, ADR-0026) and the file written by `/usr/bin/time -f "%e %M"` (wall seconds,
peak memory in KB). Columns: accuracy and recall of the five classes on the test set.
"""
import csv
import json
import sys
from pathlib import Path

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
COLUMNS = (["arm", "train_positions", "init_seed", "accuracy"]
           + ["recall_" + label.lower().replace(" ", "_") for label in LABELS]
           + ["train_seconds", "peak_rss_kb"])


def main(csv_path, arm, train_positions, init_seed, report_path, timing):
    report = json.load(open(report_path))
    by_name = {c["name"]: c for c in report["classes"]}
    seconds, peak = open(timing).read().split()
    recalls = [by_name[label]["recall"] for label in LABELS]
    accuracy = report["accuracy"]

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([arm, train_positions, init_seed, f"{accuracy:.4f}",
                    *(f"{r:.4f}" for r in recalls), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:7])
