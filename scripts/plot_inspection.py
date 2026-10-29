#!/usr/bin/env python3
"""Graphs of the network inspection (#36, scripts/inspect.sh).

Arguments: histograms.csv and saliency.csv written by scripts/inspect.cpp, the labelled test split
(for the boards) and the output directory. Writes histograms.png, saliency_examples.png,
saliency_agreement.png and saliency_agreement.csv, and prints the agreement figures.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

LABELS = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
PIECES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def board_of(fen):
    """64 characters, a8 to h1, '.' for an empty square."""
    squares = []
    for row in fen.split()[0].split("/"):
        for c in row:
            squares.extend("." * int(c) if c.isdigit() else c)
    return squares


def name(square):
    return "abcdefgh"[square % 8] + str(8 - square // 8)


def histograms(path, out):
    data = defaultdict(lambda: defaultdict(dict))  # (layer, kind) -> epoch -> bin -> count
    edges = {}
    for r in csv.DictReader(open(path, newline="")):
        key = (int(r["layer"]), r["kind"])
        data[key][int(r["epoch"])][int(r["bin"])] = int(r["count"])
        edges.setdefault(r["kind"], {})[int(r["bin"])] = (float(r["low"]), float(r["high"]))
    layers = sorted({layer for layer, _ in data})
    fig, axes = plt.subplots(2, len(layers), figsize=(5 * len(layers), 8))
    for col, layer in enumerate(layers):
        for row, kind in enumerate(["weight", "gradient"]):
            ax = axes[row][col]
            epochs = sorted(data[(layer, kind)])
            bins = sorted(edges[kind])
            grid = [[max(data[(layer, kind)][e].get(b, 0), 0.5) for e in epochs] for b in bins]
            low = edges[kind][bins[0]][0]
            high = edges[kind][bins[-1]][1]
            image = ax.imshow(grid, aspect="auto", origin="lower", norm=LogNorm(vmin=0.5),
                              extent=[epochs[0] - 0.5, epochs[-1] + 0.5, low, high],
                              cmap="viridis")
            fig.colorbar(image, ax=ax, label="number of values")
            ax.set(xlabel="epoch (0 = initial weights)",
                   ylabel="weight value" if kind == "weight" else "log10 |gradient|",
                   title=f"layer {layer} (dense), {kind}s"
                   + (" of the last batch" if kind == "gradient" else ""))
    fig.suptitle("Weight and gradient histograms per epoch, final configuration at 10 % of the data")
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=100)
    plt.close(fig)


def draw_board(ax, squares, occlusion, marks, title):
    peak = max(abs(v) for v in occlusion) or 1
    for s in range(64):
        x, y = s % 8, 7 - s // 8
        light = (x + y) % 2 == 1
        ax.add_patch(Rectangle((x, y), 1, 1, color="#f0d9b5" if light else "#b58863"))
        v = occlusion[s]
        if v:
            colour = (0.84, 0.15, 0.16) if v > 0 else (0.12, 0.47, 0.71)
            ax.add_patch(Rectangle((x, y), 1, 1, color=colour, alpha=0.8 * abs(v) / peak))
        piece = squares[s]
        if piece != ".":
            ax.text(x + 0.5, y + 0.5, piece, ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if piece.isupper() else "black")
        if s in marks:
            ax.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, fill=False, lw=2.5,
                                   ec=marks[s]))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([i + 0.5 for i in range(8)], list("abcdefgh"))
    ax.set_yticks([i + 0.5 for i in range(8)], [str(i + 1) for i in range(8)])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def saliency(path, test_file, out):
    fens = [" ".join(line.split()[:6]) for line in open(test_file)]
    rows = list(csv.DictReader(open(path, newline="")))
    groups = defaultdict(lambda: [0, 0, 0, 0])  # positions, on checker, on checker or king, gradient
    examples = {}
    for r in rows:
        index = int(r["index"])
        squares = board_of(fens[index])
        checkers = [int(c) for c in r["checkers"].split()]
        king = int(r["king"])
        right = r["expected"] == r["predicted"]
        piece = "double check" if len(checkers) > 1 else PIECES[squares[checkers[0]].lower()]
        top = int(r["top_occlusion"])
        gradient = int(r["top_gradient"])
        for key in ("all right" if right else "all wrong", piece if right else None):
            if key is None:
                continue
            g = groups[key]
            g[0] += 1
            g[1] += top in checkers
            g[2] += top in checkers or top == king
            g[3] += gradient in checkers or gradient == king
        expected = LABELS[int(r["expected"])]
        want = None
        if right and piece == "bishop" and expected.startswith("Check "):
            want = "bishop check"
        elif right and piece == "knight" and expected.startswith("Check "):
            want = "knight check"
        elif right and piece == "queen" and expected.startswith("Checkmate"):
            want = "queen mate"
        elif not right and expected.startswith("Checkmate"):
            want = "mate taken for a check"
        if want and want not in examples:
            examples[want] = (r, squares, checkers, king)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (title, (r, squares, checkers, king)) in zip(axes, examples.items()):
        occlusion = [float(r[f"o{s}"]) for s in range(64)]
        marks = {c: "#00a000" for c in checkers}
        marks[king] = "#7b2cbf"
        draw_board(ax, squares, occlusion, marks,
                   f"{title} (test position {r['index']})\nexpected {LABELS[int(r['expected'])]},"
                   f" network {LABELS[int(r['predicted'])]} {float(r['probability']):.2f}\n"
                   f"most important: {name(int(r['top_occlusion']))}")
    fig.suptitle("Probability lost when each piece is removed (red: the piece supports the answer;"
                 " green frame: checking piece, purple: king in check, from the oracle)")
    fig.tight_layout()
    fig.savefig(out / "saliency_examples.png", dpi=100)
    plt.close(fig)

    order = ["all right", "all wrong", "pawn", "knight", "bishop", "rook", "queen",
             "double check"]
    with open(out / "saliency_agreement.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "positions", "top_occlusion_on_checker",
                         "top_occlusion_on_checker_or_king", "top_gradient_on_checker_or_king"])
        for key in order:
            n, checker, either, grad = groups[key]
            writer.writerow([key, n, round(checker / n, 4), round(either / n, 4),
                             round(grad / n, 4)])
            print(f"{key}: {n} positions, most important square by occlusion on the checking "
                  f"piece {checker / n:.1%}, on it or the king {either / n:.1%}; by gradient on "
                  f"either {grad / n:.1%}")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.27
    for k, (column, label) in enumerate([(1, "occlusion: checking piece"),
                                         (2, "occlusion: checking piece or king"),
                                         (3, "gradient × input: checking piece or king")]):
        values = [groups[key][column] / groups[key][0] for key in order]
        ax.bar([i + (k - 1) * width for i in range(len(order))], values, width, label=label)
    ax.set_xticks(range(len(order)), [f"{key}\n({groups[key][0]:,})" for key in order],
                  fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of positions")
    ax.set_title("Is the most important square the checking piece or the king? (test positions in "
                 "check or mate; per piece: right answers only)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "saliency_agreement.png", dpi=100)
    plt.close(fig)


def main(histogram_csv, saliency_csv, test_file, directory):
    out = Path(directory)
    histograms(histogram_csv, out)
    saliency(saliency_csv, test_file, out)


if __name__ == "__main__":
    main(*sys.argv[1:5])
