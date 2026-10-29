#!/usr/bin/env python3
"""Analysis and graphs of the MC dropout measurement (#26, scripts/uncertainty.sh).

Arguments: the per-position CSV written by scripts/uncertainty.cpp and the output directory.
Answers "are the errors the uncertain predictions?" for five uncertainty measures: one minus the top
probability of the plain prediction (no dropout, one pass), one minus the top probability of the
MC mean, and the entropy, mutual information and disagreement of the MC passes. Writes
uncertainty.png and summary.csv.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Label indices of the network: Nothing, Check White, Check Black, Checkmate White, Checkmate Black.
SAME_SIDE = {frozenset((1, 3)), frozenset((2, 4))}
MEASURES = [
    ("plain confidence (no dropout)", "plain", lambda r: 1 - float(r["plain_confidence"])),
    ("MC confidence", "mc", lambda r: 1 - float(r["mc_confidence"])),
    ("MC entropy", "mc", lambda r: float(r["entropy"])),
    ("MC mutual information", "mc", lambda r: float(r["mutual_information"])),
    ("MC disagreement", "mc", lambda r: float(r["disagreement"])),
]


def auc(scores, errors):
    """Probability that a random error is more uncertain than a random right answer (ties 1/2)."""
    pairs = sorted(zip(scores, errors))
    positives = sum(errors)
    negatives = len(errors) - positives
    rank_sum, rank, i = 0.0, 1, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average = rank + (j - i - 1) / 2
        rank_sum += average * sum(e for _, e in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def main(csv_file, directory):
    out = Path(directory)
    rows = list(csv.DictReader(open(csv_file, newline="")))
    n = len(rows)
    summary = []
    fig, (decile_ax, coverage_ax, auc_ax) = plt.subplots(1, 3, figsize=(17, 4.8))
    for name, column, measure in MEASURES:
        scores = [measure(r) for r in rows]
        errors = [int(r["expected"] != r[column]) for r in rows]
        order = sorted(range(n), key=lambda i: scores[i])  # most certain first
        total = sum(errors)
        caught = {f: sum(errors[i] for i in order[n - int(n * f):]) / total
                  for f in (0.05, 0.1, 0.2)}
        coverage = [k / 100 for k in range(5, 101, 5)]
        accuracy = [1 - sum(errors[i] for i in order[:int(n * c)]) / int(n * c) for c in coverage]
        coverage_ax.plot([c * 100 for c in coverage], accuracy, marker="o", markersize=3,
                         label=name)
        value = auc(scores, errors)
        summary.append({
            "measure": name, "prediction": column, "accuracy": round(1 - total / n, 4),
            "errors": total, "auc": round(value, 4),
            "errors_in_most_uncertain_5pct": round(caught[0.05], 3),
            "errors_in_most_uncertain_10pct": round(caught[0.1], 3),
            "errors_in_most_uncertain_20pct": round(caught[0.2], 3),
            "accuracy_on_most_certain_90pct": round(accuracy[coverage.index(0.9)], 4),
        })
        if name in ("plain confidence (no dropout)", "MC mutual information"):
            deciles = [order[k * n // 10:(k + 1) * n // 10] for k in range(10)]
            decile_ax.plot(range(1, 11), [1 - sum(errors[i] for i in d) / len(d) for d in deciles],
                           marker="o", label=name)

    decile_ax.set(xlabel="uncertainty decile (1 = most certain tenth)", ylabel="accuracy",
                  title="Accuracy per tenth of the test split")
    coverage_ax.set(xlabel="% of positions answered (most certain first)", ylabel="accuracy",
                    title="Answering only the most certain positions")
    names = [s["measure"].replace(" (no dropout)", "\n(no dropout)").replace("MC ", "MC\n")
             for s in summary]
    bars = auc_ax.bar(names, [s["auc"] for s in summary], color="#4c78a8")
    for bar, s in zip(bars, summary):
        auc_ax.text(bar.get_x() + bar.get_width() / 2, s["auc"], f"{s['auc']:.3f}", ha="center",
                    va="bottom")
    auc_ax.set_ylim(0.5, 1)
    auc_ax.set(ylabel="AUC (errors against right answers)",
               title="How well each measure separates the errors")
    auc_ax.tick_params(axis="x", labelsize=8)
    for ax in (decile_ax, coverage_ax):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "uncertainty.png", dpi=110)

    # The main error of the network: check against checkmate of the same side.
    mc_errors = [r for r in rows if r["expected"] != r["mc"]]
    same = [r for r in mc_errors if frozenset((int(r["expected"]), int(r["mc"]))) in SAME_SIDE]
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"positions {n}, MC errors {len(mc_errors)}, check against checkmate of the same side "
          f"{len(same)} ({len(same) / len(mc_errors):.1%})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
