#!/usr/bin/env python3
"""Table and graphs of the classical baselines (#43, docs/performance/23-baselines.md), from the
directory written by scripts/bench_baselines.sh: one sub-directory per baseline (summary.csv,
scores.txt, time.txt) and ours/ (scores.txt, time.txt, bytes.txt of our networks, rescored on the same
test part and machine).
Writes DIR/baselines.csv and DIR/baselines.png. Our training times are not re-measured: they come from
docs/performance/data/final-network/*/run.txt (another machine, said in the table).
Usage: plot_baselines.py DIR"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
BASELINES = [("logreg", "Logistic regression"), ("forest", "Random forest"),
             ("mlp", "scikit-learn MLP"), ("lightgbm", "LightGBM")]
OURS = [("my_torch_network_final", "my_torch dense (pushed)", "augmented"),
        ("my_torch_network_cnn_epoch10", "my_torch CNN (epoch 10)", None)]
FINAL_RUNS = Path(__file__).resolve().parent.parent / "docs/performance/data/final-network"


def scores(path):
    text = path.read_text()
    out = {"accuracy": float(re.search(r"accuracy: ([0-9.]+)", text).group(1)),
           "macro_f1": float(re.search(r"macro F1: ([0-9.]+)", text).group(1))}
    for c in CLASSES:
        out[c] = float(re.search(rf"^{c}: .*F1 ([0-9.]+)", text, re.M).group(1))
    return out


def rows(directory):
    table = []
    for key, label in BASELINES:
        d = directory / key
        if not (d / "summary.csv").exists():
            continue
        summary = next(csv.DictReader(open(d / "summary.csv", newline="")))
        peak = int((d / "time.txt").read_text().split()[1]) / 1024
        table.append({"model": label, "kind": "baseline", "setting": summary["setting"],
                      **scores(d / "scores.txt"), "train_seconds": float(summary["train_seconds"]),
                      "train_note": "this machine, chosen setting only",
                      "predict_seconds": float(summary["predict_seconds"]),
                      "model_bytes": int(summary["model_bytes"]), "peak_mb": round(peak)})
    for key, label, run in OURS:
        d = directory / "ours" / key
        if not (d / "scores.txt").exists():
            continue
        seconds, kb = (d / "time.txt").read_text().split()
        train, note = None, "not measured for this snapshot"
        if run:
            m = re.search(r"training: ([0-9.]+) s", (FINAL_RUNS / run / "run.txt").read_text())
            train, note = float(m.group(1)), "from its run.txt, another machine"
        table.append({"model": label, "kind": "ours", "setting": key, **scores(d / "scores.txt"),
                      "train_seconds": train, "train_note": note,
                      "predict_seconds": float(seconds),
                      "model_bytes": int((d / "bytes.txt").read_text()),
                      "peak_mb": round(int(kb) / 1024)})
    return table


def main(directory):
    directory = Path(directory)
    table = rows(directory)
    fields = ["model", "kind", "setting", "accuracy", "macro_f1", *CLASSES, "train_seconds",
              "train_note", "predict_seconds", "model_bytes", "peak_mb"]
    with open(directory / "baselines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    colors = ["#e6550d" if r["kind"] == "ours" else "#3182bd" for r in table]
    names = [r["model"] for r in table]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    bars = ax.barh(names, [r["accuracy"] for r in table], color=colors)
    for bar, r in zip(bars, table):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['accuracy']:.4f} (F1 {r['macro_f1']:.4f})", va="center", fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.invert_yaxis()
    ax.set_title("Test accuracy (macro F1 in brackets), 71,555 positions", fontsize=10)

    ax = axes[0][1]
    width = 0.8 / len(table)
    for k, r in enumerate(table):
        ax.bar([i + (k - (len(table) - 1) / 2) * width for i in range(len(CLASSES))],
               [r[c] for c in CLASSES], width, label=r["model"], color=colors[k],
               alpha=0.35 + 0.65 * (k + 1) / len(table))
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test F1 per class", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1][0]
    timed = [r for r in table if r["train_seconds"]]
    bars = ax.barh([r["model"] for r in timed], [r["train_seconds"] for r in timed],
                   color=["#e6550d" if r["kind"] == "ours" else "#3182bd" for r in timed])
    for bar, r in zip(bars, timed):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['train_seconds']:,.0f} s", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_title("Training time of the chosen setting (log scale; ours on another machine)",
                 fontsize=10)

    ax = axes[1][1]
    bars = ax.barh(names, [r["model_bytes"] / 1e6 for r in table], color=colors)
    for bar, r in zip(bars, table):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {r['model_bytes'] / 1e6:,.1f} MB, predicts in {r['predict_seconds']:.1f} s",
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_title("Model file size (log scale) and prediction time of the test part", fontsize=10)

    fig.suptitle("Classical baselines (blue) against our networks (orange), same split and encoding")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(directory / "baselines.png", dpi=100)
    for r in table:
        print(f"{r['model']}: accuracy {r['accuracy']:.4f}, macro F1 {r['macro_f1']:.4f}, "
              f"train {r['train_seconds']} s, size {r['model_bytes']:,} B")


if __name__ == "__main__":
    main(sys.argv[1])
