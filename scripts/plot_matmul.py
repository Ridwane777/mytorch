#!/usr/bin/env python3
"""Draws the graphs of scripts/bench_matmul.sh: product speed and training time per thread count."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(directory):
    d = Path(directory)
    rows = list(csv.DictReader(open(d / "matmul.csv", newline="")))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    blocked = defaultdict(list)
    for r in rows:
        if r["version"] == "blocked":
            blocked[r["product"]].append((int(r["threads"]), float(r["gflops"])))
    for product, points in blocked.items():
        line, = ax1.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                         label=f"{product}, blocked")
        plain = next(float(r["gflops"]) for r in rows
                     if r["product"] == product and r["version"] == "plain loop")
        ax1.axhline(plain, color=line.get_color(), linestyle=":", linewidth=1)
    ax1.set_xlabel("threads")
    ax1.set_ylabel("GFLOP/s")
    ax1.set_title("Matrix products of a 901 → 256 layer, batch of 64\n(dotted: plain loop, 1 thread)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    train = list(csv.DictReader(open(d / "matmul_training.csv", newline="")))
    threads = [int(r["threads"]) for r in train]
    times = [float(r["seconds"]) for r in train]
    ax2.bar([str(t) for t in threads], times, color="#4f81bd")
    for i, t in enumerate(times):
        ax2.text(i, t, f"{t:.0f} s\n×{times[0] / t:.1f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("threads (MY_TORCH_THREADS)")
    ax2.set_ylabel("training time, 2 epochs (s)")
    ax2.set_title("Final network, 2 epochs on 50,000 positions")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "matmul.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
