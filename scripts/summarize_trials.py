#!/usr/bin/env python3
"""Gathers the trial runs of scripts/trial_runs.sh: one row per run in trials.csv, and trials.png.

Arguments: the runs directory (one sub-directory per run, written by scripts/train_final.sh with
EVAL=validation) and the output directory. Every score is on the validation split.
"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def value(text, pattern, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def main(runs, directory):
    rows = []
    for run in sorted(p for p in Path(runs).iterdir() if (p / "scores.txt").exists()):
        scores = (run / "scores.txt").read_text()
        card = (run / "run.txt").read_text()
        config = next(run.glob("*.conf")).read_text()
        layers = [int(m) for m in re.findall(r"^dense \d+ (\d+)", config, re.M)][:-1]
        params = 0
        sizes = [901] + layers + [5]
        for a, b in zip(sizes, sizes[1:]):
            params += a * b + b
        rows.append({
            "run": run.name,
            "hidden": "x".join(map(str, layers)),
            "init": "he_uniform" if "init=he_uniform" in config else "uniform",
            "parameters": params,
            "validation_accuracy": value(scores, r"accuracy: ([\d.]+)"),
            "validation_macro_f1": value(scores, r"macro F1: ([\d.]+)"),
            "epochs": value(card, r"epochs run: (\d+)", int),
            "train_seconds": value(card, r"training: ([\d.]+) s"),
            "peak_mb": value(card, r"training: [\d.]+ s, peak (\d+) MB", int),
        })
    rows.sort(key=lambda r: -r["validation_macro_f1"])
    d = Path(directory)
    with open(d / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    order = sorted(rows, key=lambda r: (r["parameters"], r["init"]))
    names = [f"{r['hidden']}\n{r['init']}" for r in order]
    ax1.bar(names, [100 * r["validation_macro_f1"] for r in order],
            color=["#4f81bd" if r["init"] == "uniform" else "#9bbb59" for r in order])
    ax1.set_ylabel("validation macro F1 (%)")
    ax1.set_title("Quality (validation split, never the test split)")
    low = min(100 * r["validation_macro_f1"] for r in order)
    ax1.set_ylim(max(0, low - 5), None)
    ax2.bar(names, [r["train_seconds"] / 60 for r in order], color="#c0504d")
    ax2.set_ylabel("training time (min)")
    ax2.set_title("Cost")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "trials.png", dpi=120)


if __name__ == "__main__":
    main(*sys.argv[1:3])
