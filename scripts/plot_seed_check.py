#!/usr/bin/env python3
"""Second-seed check of the trial runs (docs/performance/trials.md): gathers the five seeds of the
two best configurations (seed 1 from trials.csv, seeds 2-5 from trials-seeds/trials.csv), writes
seed_check.csv (mean, standard deviation, min, max) and seed_check.png.

Usage: plot_seed_check.py DATA_DIRECTORY
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIGS = ["c256x128_uniform", "d512x256_uniform"]


def main(directory):
    d = Path(directory)
    scores = defaultdict(list)
    for path in (d / "trials.csv", d / "trials-seeds" / "trials.csv"):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                base = row["run"].rsplit("_s", 1)[0] if row["run"].rsplit("_s", 1)[-1].isdigit() \
                    else row["run"]
                if base in CONFIGS:
                    scores[base].append((float(row["validation_macro_f1"]),
                                         float(row["validation_accuracy"])))
    with open(d / "seed_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "seeds", "macro_f1_mean", "macro_f1_stdev", "macro_f1_min",
                    "macro_f1_max", "accuracy_mean"])
        for name in CONFIGS:
            f1 = [s[0] for s in scores[name]]
            acc = [s[1] for s in scores[name]]
            w.writerow([name, len(f1), f"{statistics.mean(f1):.4f}", f"{statistics.stdev(f1):.4f}",
                        f"{min(f1):.4f}", f"{max(f1):.4f}", f"{statistics.mean(acc):.4f}"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(CONFIGS):
        f1 = [100 * s[0] for s in scores[name]]
        ax.scatter([i] * len(f1), f1, s=50, alpha=0.7, label="one seed" if i == 0 else None)
        mean = statistics.mean(f1)
        ax.hlines(mean, i - 0.2, i + 0.2, color="black", label="mean" if i == 0 else None)
    ax.set_xticks(range(len(CONFIGS)), ["901 → 256 → 128 → 5", "901 → 512 → 256 → 5"])
    ax.set_xlim(-0.6, len(CONFIGS) - 0.4)
    ax.set_ylabel("validation macro F1 (%)")
    ax.set_title("Five seeds of the two best configurations")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(d / "seed_check.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/performance/data")
