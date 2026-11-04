#!/usr/bin/env python3
"""Graph of docs/performance/24-binary-format.md: file size, save time and load time of each
network in the text format (ADR-0012) and the binary format (ADR-0042), from the CSV written by
scripts/bench_binary_format.cpp.
Usage: plot_binary_format.py binary_format.csv output.png"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"text": "#e6550d", "binary": "#3182bd"}


def main(path, out):
    rows = list(csv.DictReader(open(path, newline="")))
    networks = list(dict.fromkeys(r["network"] for r in rows))
    value = {(r["network"], r["format"]): r for r in rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (column, title, scale) in zip(
            axes, [("file_bytes", "File size (MB)", 1e6), ("save_ms", "Save time (ms)", 1),
                   ("load_ms", "Load time (ms)", 1)]):
        width = 0.38
        for k, fmt in enumerate(("text", "binary")):
            xs = [i + (k - 0.5) * width for i in range(len(networks))]
            ys = [float(value[(n, fmt)][column]) / scale for n in networks]
            bars = ax.bar(xs, ys, width, color=COLORS[fmt], label=fmt)
            for bar, y in zip(bars, ys):
                ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.2f}", ha="center",
                        va="bottom", fontsize=8)
        ax.set_xticks(range(len(networks)),
                      [f"{n}\n({int(value[(n, 'text')]['parameters']):,} parameters)"
                       for n in networks], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Network files, text format (ADR-0012) against binary format (ADR-0042); "
                 "timings: median of 9 runs")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=100)
    for n in networks:
        t, b = value[(n, "text")], value[(n, "binary")]
        print(f"{n}: {int(t['file_bytes']):,} -> {int(b['file_bytes']):,} bytes "
              f"(x{int(t['file_bytes']) / int(b['file_bytes']):.2f} smaller), load "
              f"{float(t['load_ms']):.1f} -> {float(b['load_ms']):.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
