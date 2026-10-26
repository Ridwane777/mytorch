#!/usr/bin/env python3
"""Draws the graphs of the dataset analysis from the CSV files of bonus/my_torch_eda."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(directory):
    d = Path(directory)

    # 1. Which piece gives check, as a share of the attackers of each class.
    rows = {r["label"]: r for r in read(d / "eda_checkers.csv")}
    kinds = ["pawn", "knight", "bishop", "rook", "queen"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for i, label in enumerate(CLASSES[1:]):
        total = sum(int(rows[label][k]) for k in kinds)
        ax.bar([k + (i - 1.5) * width for k in range(len(kinds))],
               [100 * int(rows[label][k]) / total for k in kinds], width, label=label)
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_ylabel("share of the checking pieces (%)")
    ax.set_title("Which piece gives check or checkmate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "eda_checkers.png", dpi=120)

    # 2. Share of positions holding at least one piece of each kind.
    pieces = defaultdict(dict)
    for r in read(d / "eda_pieces.csv"):
        pieces[r["label"]][r["piece"]] = float(r["share_with_at_least_one"])
    letters = ["Q", "R", "B", "N", "P", "q", "r", "b", "n", "p"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.16
    for i, label in enumerate(CLASSES):
        ax.bar([k + (i - 2) * width for k in range(len(letters))],
               [100 * pieces[label][l] for l in letters], width, label=label)
    ax.set_xticks(range(len(letters)), [("white " if l.isupper() else "black ") + l.upper()
                                         for l in letters], rotation=30)
    ax.set_ylabel("positions with at least one (%)")
    ax.set_title("Pieces present, per class")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(d / "eda_pieces.png", dpi=120)

    # 3. Where the king of the side to move stands (the attacked king for check and checkmate).
    kings = defaultdict(lambda: [[0] * 8 for _ in range(8)])
    for r in read(d / "eda_kings.csv"):
        sq = int(r["square"])
        kings[r["label"]][sq // 8][sq % 8] = int(r["count"])
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, label in zip(axes, CLASSES):
        grid = kings[label]
        total = sum(map(sum, grid))
        ax.imshow([[100 * v / total for v in row] for row in grid], cmap="Reds")
        ax.set_xticks(range(8), list("abcdefgh"))
        ax.set_yticks(range(8), [str(8 - r) for r in range(8)])
        edge = sum(grid[r][f] for r in range(8) for f in range(8) if r in (0, 7) or f in (0, 7))
        ax.set_title(f"{label}\n{100 * edge / total:.0f} % on the edge")
    fig.suptitle("Square of the king of the side to move (share of the class)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(d / "eda_kings.png", dpi=120)

    # 4. Material balance: side giving check minus side to move (piece values 1, 3, 3, 5, 9).
    balance = defaultdict(dict)
    for r in read(d / "eda_material.csv"):
        balance[r["label"]][int(r["attacker_minus_defender"])] = int(r["count"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label in CLASSES:
        total = sum(balance[label].values())
        xs = sorted(k for k in balance[label] if -20 <= k <= 20)
        ax.plot(xs, [100 * balance[label][x] / total for x in xs], label=label)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("material of the other side minus material of the side to move")
    ax.set_ylabel("positions (%)")
    ax.set_title("Material balance per class")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "eda_material.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
