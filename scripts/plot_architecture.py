#!/usr/bin/env python3
"""Draws a network as a vertical stack of boxes, one per layer, from the TSV that
scripts/architecture.cpp prints (docs/performance/20-cnn.md): type and settings, output shape and
number of parameters in each box, total in the title. Strictly sequential, as the library's
networks are (no branch).
Usage: plot_architecture.py ARCHITECTURE.tsv OUTPUT.png TITLE"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

COLORS = {
    "input": "#d9d9d9",
    "to_planes": "#e6d5f0",
    "board_planes": "#e6d5f0",
    "conv2d": "#9ecae1",
    "max_pool2d": "#a1d99b",
    "avg_pool2d": "#a1d99b",
    "global_avg_pool": "#a1d99b",
    "dense": "#fdd0a2",
    "softmax": "#fcae91",
}
DEFAULT_COLOR = "#f2f2f2"  # activations and the other layers that keep the shape

BOX_W = 7.4
BOX_H = 0.8
GAP = 0.45


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    layers = [r for r in rows if r["index"] != "total"]
    total = int(next(r for r in rows if r["index"] == "total")["parameters"])
    return layers, total


def settings_text(row):
    """Settings as written in the network file, without the regularization defaults."""
    words = [w for w in row["settings"].split() if w not in ("l1=0", "l2=0")]
    text = " ".join(words)
    if row["type"] == "dense" and len(words) >= 2:
        text = f"{words[0]} -> {words[1]}"
    return text


def draw(layers, total, output, title):
    n = len(layers)
    height = n * BOX_H + (n - 1) * GAP + 1.6
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * height))
    ax.set_xlim(0, BOX_W + 2.6)
    ax.set_ylim(0, height)
    ax.axis("off")
    top = height - 1.0
    for i, row in enumerate(layers):
        y = top - i * (BOX_H + GAP) - BOX_H
        kind = row["type"]
        ax.add_patch(FancyBboxPatch((0.2, y), BOX_W, BOX_H,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=COLORS.get(kind, DEFAULT_COLOR),
                                    edgecolor="#333333", linewidth=1.0))
        label = "input" if kind == "input" else f"{row['index']}. {kind}"
        detail = row["settings"] if kind == "input" else settings_text(row)
        ax.text(0.4, y + BOX_H * (0.68 if detail else 0.5), label, fontsize=10, fontweight="bold",
                va="center")
        if detail:
            ax.text(0.4, y + BOX_H * 0.27, detail, fontsize=7.5, va="center", family="monospace")
        params = int(row["parameters"])
        side = f"output {row['output']}"
        if params:
            side += f"\n{params:,} parameters"
        ax.text(BOX_W + 0.45, y + BOX_H / 2, side, fontsize=8.5, va="center")
        if i + 1 < n:
            ax.annotate("", xy=(0.2 + BOX_W / 2, y - GAP), xytext=(0.2 + BOX_W / 2, y),
                        arrowprops={"arrowstyle": "-|>", "color": "#333333", "linewidth": 1.0})
    ax.set_title(f"{title}: {total:,} parameters", fontsize=12, fontweight="bold")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    layers, total = read(sys.argv[1])
    draw(layers, total, sys.argv[2], sys.argv[3])
    print(f"{sys.argv[2]}: {len(layers) - 1} layers, {total:,} parameters")


if __name__ == "__main__":
    main()
