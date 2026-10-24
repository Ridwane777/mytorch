#!/usr/bin/env python3
"""Draws the graph of scripts/bench_serialization.cpp: file size and save/load time."""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory):
    d = Path(directory)
    with open(d / "serialization.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["network"] for r in rows]
    x = range(len(rows))
    fig, (size, time) = plt.subplots(1, 2, figsize=(11, 4))
    size.bar([i - 0.2 for i in x], [int(r["file_bytes"]) / 1e6 for r in rows], 0.4,
             label="text file (measured)")
    size.bar([i + 0.2 for i in x], [int(r["raw_bytes"]) / 1e6 for r in rows], 0.4,
             label="8 bytes per number (computed lower bound)")
    size.set_ylabel("MB")
    size.set_title("File size")
    time.bar([i - 0.2 for i in x], [float(r["save_ms"]) for r in rows], 0.4, label="save")
    time.bar([i + 0.2 for i in x], [float(r["load_ms"]) for r in rows], 0.4, label="load")
    time.set_ylabel("ms (median of 9 runs)")
    time.set_title("Save and load time")
    for ax in (size, time):
        ax.set_xticks(list(x), names)
        ax.set_xlabel("network")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(d / "serialization.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
