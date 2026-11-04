#!/usr/bin/env python3
"""Final comparison report (#44, docs/performance/26-comparison.md): gathers the measurements already
made on the same split (PyTorch reference, 22; classical baselines, 23), the rule-based oracle run
by comparison_report.sh, the current size of the network files and the lines of code of each
implementation. Re-trains nothing.
Writes OUT/comparison.csv, OUT/lines_of_code.csv and OUT/comparison.png.
Standard library and matplotlib only.
Usage: report.py ROOT ORACLE_DIR OUT"""
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]
NETWORKS = {"my_torch dense (pushed)": "my_torch_network_final.nn",
            "my_torch CNN (epoch 10)": "bonus/networks/my_torch_network_cnn_epoch10.nn"}
# What each implementation's own code is: the part a reader would have to maintain.
CODE = [
    ("my_torch library", ["include/my_torch/*.hpp", "src/my_torch/*.[ch]pp"]),
    ("my_torch library tests", ["tests/my_torch/*.cpp"]),
    ("chess analyzer (FEN, encodings, CLI)", ["src/analyzer/*.[ch]pp"]),
    ("rule-based oracle", ["bonus/oracle/*.[ch]pp"]),
    ("PyTorch twin (model + training)", ["bonus/comparaison/mytorch_net.py",
                                         "bonus/comparaison/train_reference.py"]),
    ("classical baselines", ["bonus/comparaison/baselines.py"]),
    ("Python bindings", ["bonus/python/bindings.cpp"]),
]


def code_lines(root, patterns):
    """Non-blank lines that are not only a comment (// or #), and the number of files."""
    files = sorted({p for pattern in patterns for p in root.glob(pattern)})
    count = 0
    for path in files:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("//", "#!", "# ", "#\t")) and stripped != "#":
                count += 1
    return count, len(files)


def oracle_row(directory):
    report = (directory / "oracle.txt").read_text()
    seconds, kb = (directory / "time.txt").read_text().split()
    agree = int(re.search(r"agree: (\d+)", report).group(1))
    total = int(re.search(r"labelled: (\d+)", report).group(1))
    return {"model": "Rule-based oracle", "kind": "rules", "accuracy": agree / total,
            "macro_f1": 1.0 if agree == total else None, **{c: None for c in CLASSES},
            "predict_seconds": float(seconds), "model_bytes": 0, "peak_mb": round(int(kb) / 1024),
            "train_seconds": 0, "train_note": "no training: hand-written chess rules"}


def main(root, oracle_dir, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    perf = root / "docs/performance/data"

    table = []
    for r in csv.DictReader(open(perf / "baselines/baselines.csv", newline="")):
        row = {k: r[k] for k in ("model", "kind", "train_note")}
        for k in ("accuracy", "macro_f1", *CLASSES, "predict_seconds"):
            row[k] = float(r[k])
        row["train_seconds"] = float(r["train_seconds"]) if r["train_seconds"] else None
        row["peak_mb"] = int(r["peak_mb"])
        # Our files have been binary since ADR-0042: the current size, not the text one of 23.
        name = NETWORKS.get(r["model"])
        row["model_bytes"] = (root / name).stat().st_size if name else int(r["model_bytes"])
        table.append(row)
    table.append(oracle_row(Path(oracle_dir)))

    fields = ["model", "kind", "accuracy", "macro_f1", *CLASSES, "train_seconds", "train_note",
              "predict_seconds", "model_bytes", "peak_mb"]
    with open(out / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    code = [(label, *code_lines(root, patterns)) for label, patterns in CODE]
    with open(out / "lines_of_code.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "lines", "files"])
        writer.writerows(code)

    seeds = list(csv.DictReader(open(perf / "pytorch_reference/seeds.csv", newline="")))
    summary = list(csv.DictReader(open(perf / "pytorch_reference/summary.csv", newline="")))

    colors = {"ours": "#e6550d", "baseline": "#3182bd", "rules": "#31a354"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0][0]
    ranked = sorted(table, key=lambda r: r["accuracy"])
    bars = ax.barh([r["model"] for r in ranked], [r["accuracy"] for r in ranked],
                   color=[colors[r["kind"]] for r in ranked])
    for bar, r in zip(bars, ranked):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {r['accuracy']:.4f}",
                va="center", fontsize=8)
    ax.set_xlim(0, 1.15)
    ax.set_title("Test accuracy, 71,555 positions (green: rules, orange: ours, blue: baselines)",
                 fontsize=10)

    ax = axes[0][1]
    pairs = [("dense", "my_torch", "my_torch"), ("dense", "torch_native_float64", "PyTorch"),
             ("cnn", "my_torch", "my_torch"), ("cnn", "torch_native_float64", "PyTorch")]
    labels, values, errors, tints = [], [], [], []
    for model, variant, name in pairs:
        r = next(s for s in seeds if s["model"] == model and s["variant"] == variant)
        labels.append(f"{model}\n{name}\n{r['seeds']} seeds")
        values.append(float(r["val_accuracy_mean"]))
        errors.append(float(r["val_accuracy_std"] or 0))
        tints.append(colors["ours"] if variant == "my_torch" else "#756bb1")
    ax.bar(labels, values, yerr=errors, capsize=4, color=tints)
    for k, v in enumerate(values):
        ax.text(k, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Same architecture and budget (10 epochs): validation accuracy, mean ± std",
                 fontsize=10)

    ax = axes[1][0]
    epoch = {}
    for model in ("dense", "cnn"):
        for variant, name in (("my_torch", "my_torch"), ("torch_native_float64", "PyTorch f64"),
                              ("torch_native_float32", "PyTorch f32")):
            runs = [float(s["median_epoch_s"]) for s in summary
                    if s["model"] == model and s["variant"] == variant]
            if runs:
                epoch[f"{model} {name}"] = sorted(runs)[len(runs) // 2]
    bars = ax.barh(list(epoch), list(epoch.values()),
                   color=[colors["ours"] if "my_torch" in k else "#756bb1" for k in epoch])
    for bar, v in zip(bars, epoch.values()):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {v:.1f} s", va="center",
                fontsize=8)
    ax.set_xlim(0, max(epoch.values()) * 1.15)
    ax.invert_yaxis()
    ax.set_title("Seconds per training epoch (57,244 positions), median over seeds", fontsize=10)

    ax = axes[1][1]
    bars = ax.barh([c[0] for c in code], [c[1] for c in code], color="#636363")
    for bar, c in zip(bars, code):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {c[1]:,} ({c[2]} files)",
                va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Lines of code (non-blank, not comment-only)", fontsize=10)

    fig.suptitle("my_torch against PyTorch, classical methods and the rule-based oracle")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "comparison.png", dpi=100)
    for r in table:
        print(f"{r['model']}: accuracy {r['accuracy']:.4f}, file {r['model_bytes']:,} B")
    for label, lines, files in code:
        print(f"{label}: {lines} lines in {files} files")
    print("epoch seconds:", {k: round(v, 1) for k, v in epoch.items()})


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
