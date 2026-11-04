"""Trains the PyTorch twin of a my_torch configuration (#42, ADR-0040).

Same data (the bytes written by `pytorch_reference export`, the library's own encoding), same
architecture (read from the configuration), same loss, optimizer (Adam, same learning rate and
epsilon), batch size and number of epochs as `pytorch_reference train`, which trains our side.
Two choices say what is held equal beyond that:

  --init shared    start from the exact initial weights my_torch drew (DIR/init_weights.bin);
  --init native    PyTorch's own default initialisation, seeded with the configuration's init seed;
  --order shared   replay the batches in the order my_torch used (DIR/orders.bin);
  --order native   shuffle with torch.randperm, seeded with the configuration's train seed.
--seed-index K adds K to both seeds (my_torch: `pytorch_reference train ... INDEX`), to measure the
spread between seeds.

With `--init shared --order shared` in float64 the two programs start from the same point and see the
same batches: what is left to differ is floating-point rounding, and equivalence.csv measures it
against my_torch's results in DIR.

Usage: python train_reference.py --config C --train PREFIX --valid PREFIX --epochs N
           --variant NAME --model NAME --dtype float64|float32 --init shared|native
           --order shared|native --threads T --out-dir OUT [--reference DIR]
Writes OUT/epochs.csv (epoch, seconds of the training pass only, train loss and accuracy as measured
on the batches while training, validation loss and accuracy after the epoch), OUT/run.txt (memory
and timing) and, with --reference, OUT/equivalence.csv.
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mytorch_net


def rss_mb() -> float:
    """Peak resident set size of this process so far, in MB (1 MB = 10^6 bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6


RSS_AFTER_IMPORT = rss_mb()


def evaluate(net, x: np.ndarray, y: np.ndarray, dtype) -> tuple[float, float, np.ndarray]:
    """Mean cross-entropy, accuracy and predicted classes on a whole set, 4096 rows at a time."""
    total, right, predicted = 0.0, 0, []
    with torch.no_grad():
        for begin in range(0, len(y), 4096):
            xb = torch.from_numpy(x[begin : begin + 4096]).to(dtype)
            yb = torch.from_numpy(y[begin : begin + 4096]).long()
            scores = net(xb)
            total += float(F.cross_entropy(scores, yb, reduction="sum"))
            classes = scores.argmax(dim=1)
            right += int((classes == yb).sum())
            predicted.append(classes.numpy().astype(np.uint8))
    return total / len(y), right / len(y), np.concatenate(predicted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--init", choices=["shared", "native"], required=True)
    ap.add_argument("--order", choices=["shared", "native"], required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--reference")
    ap.add_argument("--seed-index", type=int, default=0, help="added to the init and shuffle seeds")
    args = ap.parse_args()
    if (args.init == "shared" or args.order == "shared") and not args.reference:
        ap.error("--reference DIR is needed for --init shared and --order shared")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    dtype = getattr(torch, args.dtype)
    config = mytorch_net.read_config(args.config)
    batch = int(config.train["batch"])
    train_seed = int(config.train["seed"]) + args.seed_index
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x, y = mytorch_net.read_bytes(args.train, config.inputs)
    vx, vy = mytorch_net.read_bytes(args.valid, config.inputs)
    rows = len(y)
    input_mb = (x.nbytes + vx.nbytes) / 1e6
    rss_loaded = rss_mb()

    torch.manual_seed(config.init_seed + args.seed_index)
    net = mytorch_net.Net(config).to(dtype)
    reference = Path(args.reference) if args.reference else None
    manifest = None
    if reference:
        manifest = mytorch_net.read_manifest(reference / "manifest.txt")
        mytorch_net.check_manifest(net, manifest)
    if args.init == "shared":
        mytorch_net.load_parameters(net, mytorch_net.read_parameters(reference / "init_weights.bin", manifest))

    if config.optimizer != "adam":
        raise ValueError("only the adam optimizer is reproduced")
    p = config.optimizer_params
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(p.get("lr", 0.001)),
        betas=(float(p.get("beta1", 0.9)), float(p.get("beta2", 0.999))),
        eps=float(p.get("epsilon", 1e-8)),
    )

    if args.order == "shared":
        orders = np.fromfile(reference / "orders.bin", dtype="<u4")
        if orders.size < args.epochs * rows:
            raise ValueError("orders.bin holds fewer epochs than requested")
        orders = orders[: args.epochs * rows].reshape(args.epochs, rows).astype(np.int64)
    else:
        generator = torch.Generator().manual_seed(train_seed)
        orders = None

    history = []
    for epoch in range(args.epochs):
        order = orders[epoch] if orders is not None else torch.randperm(rows, generator=generator).numpy()
        net.train()
        loss_sum, correct = 0.0, 0
        start = time.perf_counter()
        for begin in range(0, rows, batch):
            index = order[begin : begin + batch]
            xb = torch.from_numpy(x[index]).to(dtype)
            yb = torch.from_numpy(y[index]).long()
            optimizer.zero_grad()
            scores = net(xb)
            loss = F.cross_entropy(scores, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(index)
            correct += int((scores.argmax(dim=1) == yb).sum())
        seconds = time.perf_counter() - start
        net.eval()
        val_loss, val_accuracy, predicted = evaluate(net, vx, vy, dtype)
        history.append((epoch + 1, seconds, loss_sum / rows, correct / rows, val_loss, val_accuracy))
        print(
            f"{args.model} {args.variant} epoch {epoch + 1}: {seconds:.2f} s, train loss "
            f"{loss_sum / rows:.4f}, validation accuracy {val_accuracy:.4f}",
            file=sys.stderr,
        )

    with open(out / "epochs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "seconds", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for row in history:
            writer.writerow([row[0]] + [f"{v:.17g}" for v in row[1:]])
    with open(out / "run.txt", "w") as f:
        f.write(f"torch {torch.__version__}, numpy {np.__version__}, threads {args.threads}\n")
        f.write(f"rss_after_import_mb {RSS_AFTER_IMPORT:.1f}\n")
        f.write(f"rss_after_data_mb {rss_loaded:.1f}\n")
        f.write(f"rss_peak_mb {rss_mb():.1f}\n")
        f.write(f"input_bytes_mb {input_mb:.1f}\n")

    if reference and args.init == "shared" and args.order == "shared":
        write_equivalence(out, args, net, manifest, reference, history, predicted)
    return 0


def write_equivalence(out, args, net, manifest, reference, history, predicted) -> None:
    """How far the end of this run is from my_torch's (same start, same batches)."""
    ours = mytorch_net.read_parameters(reference / "final_weights.bin", manifest)
    abs_diff, squares, norms = 0.0, 0.0, 0.0
    for entry, a in zip(net.entries, ours):
        b = entry.to_mytorch(entry.tensor)
        abs_diff = max(abs_diff, float(np.abs(a - b).max()))
        squares += float(((a - b) ** 2).sum())
        norms += float((a**2).sum())
    theirs = np.fromfile(reference / "valid_predictions.u8", dtype=np.uint8)
    agreement = float((theirs == predicted).mean())
    with open(reference / "epochs.csv") as f:
        mine = list(csv.DictReader(f))[: len(history)]
    d_train = max(abs(float(m["train_loss"]) - h[2]) for m, h in zip(mine, history))
    d_val = max(abs(float(m["val_loss"]) - h[4]) for m, h in zip(mine, history))
    d_acc = max(abs(float(m["val_accuracy"]) - h[5]) for m, h in zip(mine, history))
    with open(out / "equivalence.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "variant",
                "epochs",
                "max_abs_weight_diff",
                "l2_rel_weight_diff",
                "prediction_agreement",
                "max_abs_train_loss_diff",
                "max_abs_val_loss_diff",
                "max_abs_val_accuracy_diff",
            ]
        )
        writer.writerow(
            [
                args.model,
                args.variant,
                len(history),
                f"{abs_diff:.3e}",
                f"{(squares / norms) ** 0.5:.3e}",
                f"{agreement:.6f}",
                f"{d_train:.3e}",
                f"{d_val:.3e}",
                f"{d_acc:.3e}",
            ]
        )


if __name__ == "__main__":
    sys.exit(main())
