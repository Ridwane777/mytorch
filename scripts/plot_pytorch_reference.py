#!/usr/bin/env python3
"""Tables and graphs of scripts/bench_pytorch_reference.sh (docs/performance/22-pytorch-reference.md,
ADR-0040). Reads CSV and text files only: it does not import PyTorch (ADR-0036), and the scores it
tabulates come from the training programs (scripts/pytorch_reference.cpp, comparison/train_reference.py).

Usage:
  plot_pytorch_reference.py gradient RUN OUT MODEL...   gradient_check.csv, gradient_check.png
  plot_pytorch_reference.py sweep RUN OUT MODEL...      thread_sweep.csv, thread_sweep.png
  plot_pytorch_reference.py best-threads CSV MODEL IMPL prints the thread count with the fewest
                                                        seconds per epoch in thread_sweep.csv
  plot_pytorch_reference.py collect RUN OUT MODEL...    epochs.csv, summary.csv, seeds.csv,
                                                        equivalence.csv, curves.png, time_memory.png,
                                                        accuracy_seeds.png
RUN is the large run directory (runs/pytorch_reference), OUT the directory of the committed results.
"""
import csv
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOLERANCE = 1e-5
NAMES = {"dense": "Dense 901-256-128-5", "cnn": "CNN (2 convolutions)"}
VARIANTS = [
    ("my_torch", "my_torch (float64)", "#e6550d"),
    ("torch_shared_float64", "PyTorch float64, our weights and batches", "#31a354"),
    ("torch_native_float64", "PyTorch float64, own init and shuffle", "#3182bd"),
    ("torch_native_float32", "PyTorch float32, own init and shuffle", "#756bb1"),
]
LABEL = {key: label for key, label, _ in VARIANTS}
COLOR = {key: color for key, _, color in VARIANTS}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gradient(run, out, models):
    rows = []
    for model in models:
        for state in ("init", "warm"):
            rows += read_csv(Path(run) / "gradient" / f"{model}_{state}" / "check.csv")
    write_csv(Path(out) / "gradient_check.csv", rows)

    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    for ax, model in zip(axes[0], models):
        names = [r["quantity"] for r in rows if r["model"] == f"{model}_init" and r["dtype"] == "float64"]
        series = [
            ("float64, initial weights", f"{model}_init", "float64", "#31a354"),
            ("float64, after 100 Adam steps", f"{model}_warm", "float64", "#006d2c"),
            ("float32, initial weights", f"{model}_init", "float32", "#fd8d3c"),
        ]
        width = 0.27
        for k, (label, tag, dtype, color) in enumerate(series):
            values = [
                max(float(r["rel_to_max"]), 1e-18)
                for q in names
                for r in rows
                if r["model"] == tag and r["dtype"] == dtype and r["quantity"] == q
            ]
            ax.bar([i + (k - 1) * width for i in range(len(names))], values, width, label=label, color=color)
        ax.axhline(TOLERANCE, color="black", linestyle="--", linewidth=1, label="1e-5 (criterion)")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1e-3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n", 1) for n in names], fontsize=7, rotation=60)
        ax.set_ylabel("max |ours - autograd| / max |autograd|")
        ax.set_title(f"{NAMES[model]}: one batch of 64 positions")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out) / "gradient_check.png", dpi=130)


def epoch_rows(path):
    return read_csv(path)


def sweep_seconds(path):
    """Mean seconds of the epochs after the first (the first one pays the allocations)."""
    seconds = [float(r["seconds"]) for r in epoch_rows(path)]
    return statistics.mean(seconds[1:] if len(seconds) > 1 else seconds)


def sweep(run, out, models):
    rows = []
    for model in models:
        for d in sorted((Path(run) / "sweep" / model).iterdir()):
            impl, threads = d.name.rsplit("_", 1)
            rows.append(
                {
                    "model": model,
                    "implementation": impl,
                    "threads": int(threads),
                    "seconds_per_epoch": f"{sweep_seconds(d / 'epochs.csv'):.4f}",
                }
            )
    rows.sort(key=lambda r: (r["model"], r["implementation"], r["threads"]))
    write_csv(Path(out) / "thread_sweep.csv", rows)

    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.2), squeeze=False)
    colors = {"my_torch": COLOR["my_torch"], "torch_float64": COLOR["torch_native_float64"],
              "torch_float32": COLOR["torch_native_float32"]}
    for ax, model in zip(axes[0], models):
        for impl, color in colors.items():
            pts = [(r["threads"], float(r["seconds_per_epoch"])) for r in rows
                   if r["model"] == model and r["implementation"] == impl]
            ax.plot(*zip(*pts), marker="o", color=color, label=impl)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("threads")
        ax.set_ylabel("seconds per epoch")
        ax.set_title(f"{NAMES[model]}: thread scaling")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(out) / "thread_sweep.png", dpi=130)


def best_threads(csv_path, model, impl):
    rows = [r for r in read_csv(csv_path) if r["model"] == model and r["implementation"] == impl]
    print(min(rows, key=lambda r: float(r["seconds_per_epoch"]))["threads"])


def read_run(d):
    """Everything one training directory holds."""
    wall, peak_kb = Path(d, "time.txt").read_text().split()[-2:]
    info = {"wall_s": float(wall), "peak_rss_mb": int(peak_kb) / 1024}
    info["threads"] = int(Path(d, "threads.txt").read_text().split()[1])
    info["load1"] = float(Path(d, "load.txt").read_text().split()[0])
    info["epochs"] = epoch_rows(Path(d, "epochs.csv"))
    info["import_mb"] = info["input_mb"] = ""
    if Path(d, "run.txt").exists():  # written by comparison/train_reference.py only
        values = dict(line.split()[-2:] for line in Path(d, "run.txt").read_text().splitlines()[1:])
        info["import_mb"] = values["rss_after_import_mb"]
        info["input_mb"] = values["input_bytes_mb"]
    return info


def collect(run, out, models):
    out = Path(out)
    train_rows = sum(1 for _ in open(Path(run) / "data" / "train.txt"))
    valid_rows = sum(1 for _ in open(Path(run) / "data" / "valid.txt"))
    epochs_out, summary, equivalence = [], [], []
    for model in models:
        for d in sorted((Path(run) / "train" / model).iterdir()):
            variant, seed = d.name.rsplit("_s", 1)
            info = read_run(d)
            seconds = [float(r["seconds"]) for r in info["epochs"]]
            last = info["epochs"][-1]
            for r in info["epochs"]:
                epochs_out.append(
                    {"model": model, "variant": variant, "seed": seed, "epoch": r["epoch"],
                     "seconds": f"{float(r['seconds']):.4f}", "train_loss": f"{float(r['train_loss']):.6f}",
                     "train_accuracy": f"{float(r['train_accuracy']):.6f}",
                     "val_loss": f"{float(r['val_loss']):.6f}", "val_accuracy": f"{float(r['val_accuracy']):.6f}"}
                )
            summary.append(
                {
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "threads": info["threads"],
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                    "epochs": len(seconds),
                    "median_epoch_s": f"{statistics.median(seconds):.3f}",
                    "min_epoch_s": f"{min(seconds):.3f}",
                    "max_epoch_s": f"{max(seconds):.3f}",
                    "train_s": f"{sum(seconds):.1f}",
                    "wall_s": f"{info['wall_s']:.1f}",
                    "peak_rss_mb": f"{info['peak_rss_mb']:.0f}",
                    "rss_after_import_mb": info["import_mb"],
                    "input_bytes_mb": info["input_mb"],
                    "load1_at_start": f"{info['load1']:.2f}",
                    "final_train_loss": f"{float(last['train_loss']):.4f}",
                    "final_val_loss": f"{float(last['val_loss']):.4f}",
                    "final_val_accuracy": f"{float(last['val_accuracy']):.4f}",
                }
            )
            if (d / "equivalence.csv").exists():
                equivalence += read_csv(d / "equivalence.csv")
    write_csv(out / "epochs.csv", epochs_out)
    write_csv(out / "summary.csv", summary)
    if equivalence:
        write_csv(out / "equivalence.csv", equivalence)

    seeds = []
    for model in models:
        for variant, _, _ in VARIANTS:
            rows = [r for r in summary if r["model"] == model and r["variant"] == variant]
            if not rows:
                continue
            acc = [float(r["final_val_accuracy"]) for r in rows]
            loss = [float(r["final_val_loss"]) for r in rows]
            seeds.append(
                {
                    "model": model,
                    "variant": variant,
                    "seeds": len(rows),
                    "val_accuracy_mean": f"{statistics.mean(acc):.4f}",
                    "val_accuracy_std": f"{statistics.stdev(acc):.4f}" if len(acc) > 1 else "",
                    "val_accuracy_min": f"{min(acc):.4f}",
                    "val_accuracy_max": f"{max(acc):.4f}",
                    "val_loss_mean": f"{statistics.mean(loss):.4f}",
                    "val_loss_std": f"{statistics.stdev(loss):.4f}" if len(loss) > 1 else "",
                }
            )
    write_csv(out / "seeds.csv", seeds)

    curves(epochs_out, out, models)
    time_memory(summary, out, models)
    accuracy(summary, out, models)


def curves(rows, out, models):
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4.2 * len(models)), squeeze=False)
    for r, model in enumerate(models):
        for c, (key, title) in enumerate(
            [("train_loss", "training loss (batches as trained)"), ("val_loss", "validation loss"),
             ("val_accuracy", "validation accuracy")]
        ):
            ax = axes[r][c]
            for variant, label, color in VARIANTS:
                pts = [(int(x["epoch"]), float(x[key])) for x in rows
                       if x["model"] == model and x["variant"] == variant and x["seed"] == "0"]
                if pts:
                    ax.plot(*zip(*pts), marker="o", markersize=3, color=color, label=label,
                            linestyle="--" if variant == "torch_shared_float64" else "-")
            ax.set_xlabel("epoch")
            ax.set_title(f"{NAMES[model]}: {title}", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)


def time_memory(summary, out, models):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    width = 0.2
    for k, (variant, label, color) in enumerate(VARIANTS):
        for ax, key, err in ((axes[0], "median_epoch_s", True), (axes[1], "peak_rss_mb", False)):
            xs, ys, lows, highs = [], [], [], []
            for i, model in enumerate(models):
                rows = [r for r in summary if r["model"] == model and r["variant"] == variant and r["seed"] == "0"]
                if rows:
                    r = rows[0]
                    xs.append(i + (k - 1.5) * width)
                    ys.append(float(r[key]))
                    lows.append(float(r[key]) - float(r["min_epoch_s"]) if err else 0)
                    highs.append(float(r["max_epoch_s"]) - float(r[key]) if err else 0)
            ax.bar(xs, ys, width, color=color, label=label,
                   yerr=[lows, highs] if err else None, capsize=2)
    for ax, title in ((axes[0], "median seconds per epoch (bars: min and max epoch)"),
                      (axes[1], "peak resident memory of the process (MB)")):
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([NAMES[m] for m in models])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "time_memory.png", dpi=130)


def accuracy(summary, out, models):
    fig, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 4.4), squeeze=False)
    for ax, model in zip(axes[0], models):
        for i, (variant, label, color) in enumerate(VARIANTS):
            values = [float(r["final_val_accuracy"]) for r in summary
                      if r["model"] == model and r["variant"] == variant]
            if values:
                ax.scatter([i] * len(values), values, color=color, s=28, zorder=3)
                ax.hlines(statistics.mean(values), i - 0.25, i + 0.25, color=color, linewidth=2)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(["my_torch", "PyTorch\nour weights\nand batches", "PyTorch f64\nown init", "PyTorch f32\nown init"],
                           fontsize=8)
        ax.set_ylabel("validation accuracy after the last epoch")
        ax.set_title(f"{NAMES[model]}: one point per seed, bar = mean", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_seeds.png", dpi=130)


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    mode, rest = argv[0], argv[1:]
    if mode == "best-threads":
        best_threads(*rest)
    elif mode in ("gradient", "sweep", "collect"):
        {"gradient": gradient, "sweep": sweep, "collect": collect}[mode](rest[0], rest[1], rest[2:])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
