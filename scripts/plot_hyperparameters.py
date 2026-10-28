#!/usr/bin/env python3
"""Draws the graphs of scripts/hyperparam_search.sh (docs/performance/15-hyperparameters.md). The
scores are the validation macro F1 computed by my_torch (ADR-0026); this script only reads and draws.

Arguments: the directory holding grid.csv and, when they exist, random_SEED.csv and seeds.csv; the
random seed. Writes hyperparameters.png (every combination, ranked; best score found against the
number of trials for the grid and the random search; mean score per value of each hyperparameter)
and, with seeds.csv, hyperparameters_seeds.png (finalists, mean and spread over the seeds).
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT = {"lr": "0.001", "batch": "64", "hidden": "256x128", "dropout": "0"}
PARAMS = ["lr", "batch", "hidden", "dropout"]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label(r):
    return f"lr {r['lr']}, batch {r['batch']}, {r['hidden']}, dropout {r['dropout']}"


def best_so_far(scores):
    out, best = [], float("-inf")
    for s in scores:
        best = max(best, s)
        out.append(best)
    return out


def main(directory, random_seed):
    d = Path(directory)
    grid = rows(d / "grid.csv")
    for r in grid:
        r["score"] = float(r["score"])
    top = max(r["score"] for r in grid)
    low = min(r["score"] for r in grid)
    fig = plt.figure(figsize=(14, 10))

    ranked = sorted(grid, key=lambda r: r["score"])
    ax = fig.add_subplot(2, 2, (1, 3))
    colours = ["#c0504d" if all(r[p] == CURRENT[p] for p in PARAMS) else
               ("#4f81bd" if r["hidden"] == "256x128" else "#9bbb59") for r in ranked]
    ax.barh(range(len(ranked)), [r["score"] for r in ranked], color=colours)
    ax.set_yticks(range(len(ranked)), [label(r) for r in ranked], fontsize=7)
    ax.set_xlim(low - 0.02, top + 0.01)
    ax.set_xlabel("validation macro F1")
    ax.set_title("Every combination (blue 256x128, green 512x256, red: current configuration)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(range(1, len(grid) + 1), best_so_far([r["score"] for r in grid]), marker=".",
            label="grid, in grid order")
    random_path = d / f"random_{random_seed}.csv"
    if random_path.exists():
        drawn = [float(r["score"]) for r in rows(random_path)]
        ax.plot(range(1, len(drawn) + 1), best_so_far(drawn), marker="o",
                label=f"random search, seed {random_seed}")
    ax.axhline(top, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("number of trials")
    ax.set_ylabel("best validation macro F1 so far")
    ax.set_title("Best score found against the compute budget")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    width = 0.8 / 3
    for k, p in enumerate(PARAMS):
        by_value = defaultdict(list)
        for r in grid:
            by_value[r[p]].append(r["score"])
        values = list(by_value)
        means = [statistics.mean(by_value[v]) for v in values]
        xs = [k + (i - (len(values) - 1) / 2) * width for i in range(len(values))]
        ax.bar(xs, means, width * 0.9, color="#4f81bd")
        for x, m, v in zip(xs, means, values):
            ax.text(x, m, f" {v}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(range(len(PARAMS)), PARAMS)
    ax.set_ylim(low - 0.01, top + 0.02)
    ax.set_ylabel("mean validation macro F1 over the grid")
    ax.set_title("Mean score per value of each hyperparameter")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "hyperparameters.png", dpi=120)

    seeds_path = d / "seeds.csv"
    if seeds_path.exists():
        by_trial = defaultdict(list)
        for r in grid:
            by_trial[r["index"]].append(r["score"])
        seeds = rows(seeds_path)
        for r in seeds:
            by_trial[r["trial"]].append(float(r["score"]))
        finalists = list(dict.fromkeys(r["trial"] for r in seeds))
        names = {r["index"]: label(r) for r in grid}
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, t in enumerate(finalists):
            scores = by_trial[t]
            ax.scatter([i] * len(scores), scores, color="grey", zorder=3)
            ax.errorbar(i, statistics.mean(scores), yerr=statistics.stdev(scores), fmt="o",
                        color="#c0504d", capsize=6, zorder=4)
        ax.set_xticks(range(len(finalists)), [names[t].replace(", ", "\n") for t in finalists],
                      fontsize=8)
        ax.set_ylabel("validation macro F1")
        ax.set_title("Finalists over several training seeds (grey: each seed; red: mean ± sd)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(d / "hyperparameters_seeds.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "7")
