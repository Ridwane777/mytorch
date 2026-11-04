#!/usr/bin/env python3
"""Graph of the Python bindings benchmark (ADR-0044, docs/performance/25-python-bindings.md),
from DIR/times.csv written by scripts/bench_python_bindings.sh. Writes DIR/python_bindings.png.
Usage: plot_python_bindings.py DIR"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory):
    directory = Path(directory)
    times = {"cpp": [], "python": []}
    for row in csv.DictReader(open(directory / "times.csv", newline="")):
        times[row["side"]].append(float(row["seconds"]))
    names = {"cpp": "C++ (Network::predict)", "python": "Python (module, with NumPy copies)"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, side in enumerate(times):
        ax.scatter([k] * len(times[side]), times[side], color="#3182bd", alpha=0.6, zorder=3)
        median = statistics.median(times[side])
        ax.hlines(median, k - 0.25, k + 0.25, color="#e6550d", linewidth=2, zorder=4)
        ax.text(k + 0.28, median, f"median {median:.3f} s", va="center", fontsize=9)
    ax.set_xticks(range(len(times)), [names[s] for s in times])
    ax.set_xlim(-0.5, len(times) - 0.2)
    ax.set_ylim(0, max(max(t) for t in times.values()) * 1.2)
    ax.set_ylabel("seconds")
    ax.set_title("Prediction of the 71,555 test positions, same network and machine (dots: runs)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(directory / "python_bindings.png", dpi=100)


if __name__ == "__main__":
    main(sys.argv[1])
