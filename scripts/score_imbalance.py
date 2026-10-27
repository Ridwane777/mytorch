#!/usr/bin/env python3
"""Scores one run of scripts/bench_imbalance.sh and appends a row to imbalance_runs.csv.

Arguments: the CSV, the option name (none, weighted, oversample, undersample), the index of the
initial weights (1 to 3), the number of positions of the training file of the option, the
report.json and families.json of scripts/score.cpp --families for the test set (the metrics come
from my_torch, ADR-0026) and the file written by `/usr/bin/time -f "%e %M"` (wall seconds, peak
memory in KB).
Columns: accuracy, macro F1, F1 of each of the five classes and F1 of the three families
(Nothing, Check = both colours, Checkmate = both colours) on the test set.
"""
import csv
import json
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


def main(csv_path, option, init_seed, train_size, report_path, families_path, timing):
    report = json.load(open(report_path))
    families = {c["name"]: c for c in json.load(open(families_path))["classes"]}
    by_name = {c["name"]: c for c in report["classes"]}
    seconds, peak = open(timing).read().split()
    f1s = [by_name[label]["f1"] for label in LABELS]
    family_f1s = [families[name]["f1"] for name in FAMILIES]
    accuracy = report["accuracy"]
    macro_f1 = report["macro"]["f1"]

    new = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([option, init_seed, train_size, f"{accuracy:.4f}", f"{macro_f1:.4f}",
                    *(f"{v:.4f}" for v in f1s), *(f"{v:.4f}" for v in family_f1s), seconds, peak])


if __name__ == "__main__":
    main(*sys.argv[1:8])
