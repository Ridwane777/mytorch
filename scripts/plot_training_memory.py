#!/usr/bin/env python3
"""Draws the peak memory of scripts/bench_training_memory.sh (training_memory.csv)."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "before_768": "before\n(whole file encoded)",
    "after_768": "after\n(batch by batch)",
    "after_901_full": "after, full dataset\nfen-attacks-901",
}


def main(directory):
    d = Path(directory)
    with open(d / "training_memory.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [LABELS.get(r["run"], r["run"]) + f"\n{int(r['positions']):,} positions" for r in rows]
    values = [int(r["peak_kb"]) / 1024 for r in rows]
    bars = ax.bar(names, values, color=["#c0504d", "#4f81bd", "#4f81bd"][: len(rows)])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f} MB", ha="center",
                va="bottom")
    ax.set_ylabel("peak memory of --train, one epoch (MB)")
    ax.set_title("Training memory before and after the data source")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "training_memory.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
