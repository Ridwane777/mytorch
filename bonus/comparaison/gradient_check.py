"""Our gradients against PyTorch's autograd, on one batch with the same weights (#42, ADR-0040).

Input: the directory written by `pytorch_reference gradient CONFIG DATA ROWS DIR` (scripts/
pytorch_reference.cpp): the weights my_torch starts from, the gradients it computes on its first
ROWS positions, its loss and its output probabilities. The same weights are loaded into the PyTorch
twin of the configuration, which computes the same batch with autograd.

Per parameter tensor, with `ref` the autograd gradient and `got` ours:
  max_abs_diff = max |got - ref|
  rel_to_max   = max_abs_diff / max |ref|        (error relative to the largest gradient of the tensor)
  l2_rel       = ||got - ref|| / ||ref||
A tensor passes when max_abs_diff <= TOL and rel_to_max <= TOL (TOL = 1e-5, the criterion of the
issue). Run in float64 (my_torch's Scalar, the gating check) and in float32 (PyTorch's default
dtype, reported for information: it shows what single precision alone would do to the criterion).

Usage: python gradient_check.py CONFIG DIR OUT_CSV [MODEL_NAME]
Exit code 0 when every float64 tensor, the loss and the probabilities are within TOL, else 1.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import mytorch_net

TOL = 1e-5


def check(config, directory: Path, dtype: torch.dtype, rows: list[dict], model: str) -> bool:
    net = mytorch_net.Net(config).to(dtype)
    manifest = mytorch_net.read_manifest(directory / "manifest.txt")
    mytorch_net.check_manifest(net, manifest)
    mytorch_net.load_parameters(net, mytorch_net.read_parameters(directory / "weights.bin", manifest))
    ours = mytorch_net.read_parameters(directory / "gradients.bin", manifest)
    x, y = mytorch_net.read_bytes(directory / "batch", config.inputs)

    scores = net(torch.from_numpy(x).to(dtype))
    loss = F.cross_entropy(scores, torch.from_numpy(y).long())
    loss.backward()
    name = str(dtype).removeprefix("torch.")

    ok = True
    expected_probabilities = np.fromfile(directory / "probabilities.bin", dtype="<f8").reshape(len(y), -1)
    probabilities = torch.softmax(scores.detach().double(), dim=1).numpy()
    my_loss = float(Path(directory / "loss.txt").read_text())
    stats = [
        ("loss", 1, 1, abs(my_loss - float(loss.detach())), abs(float(loss.detach())), abs(my_loss - float(loss.detach())), abs(float(loss.detach()))),
        (
            "probabilities",
            *expected_probabilities.shape,
            float(np.abs(probabilities - expected_probabilities).max()),
            float(np.abs(probabilities).max()),
            float(np.linalg.norm(probabilities - expected_probabilities)),
            float(np.linalg.norm(probabilities)),
        ),
    ]
    for entry, (layer, kind, index, r, c), got in zip(net.entries, manifest, ours):
        ref = entry.to_mytorch(entry.tensor.grad)
        diff = np.abs(got - ref)
        stats.append(
            (
                f"layer{layer}_{kind}_{'weights' if index == 0 else 'bias'}",
                r,
                c,
                float(diff.max()),
                float(np.abs(ref).max()),
                float(np.linalg.norm(got - ref)),
                float(np.linalg.norm(ref)),
            )
        )
    for label, r, c, max_abs, ref_max, l2_diff, l2_ref in stats:
        rel_to_max = max_abs / ref_max if ref_max > 0 else float("inf" if max_abs > 0 else 0)
        l2_rel = l2_diff / l2_ref if l2_ref else float("nan")
        passed = max_abs <= TOL and rel_to_max <= TOL
        if dtype == torch.float64:
            ok = ok and passed
        rows.append(
            {
                "model": model,
                "dtype": name,
                "quantity": label,
                "rows": r,
                "cols": c,
                "max_abs_diff": f"{max_abs:.3e}",
                "max_abs_reference": f"{ref_max:.3e}",
                "rel_to_max": f"{rel_to_max:.3e}",
                "l2_rel": f"{l2_rel:.3e}",
                "pass": int(passed),
            }
        )
    return ok


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    config = mytorch_net.read_config(argv[0])
    model = argv[3] if len(argv) == 4 else Path(argv[0]).stem
    rows: list[dict] = []
    ok = check(config, Path(argv[1]), torch.float64, rows, model)
    check(config, Path(argv[1]), torch.float32, rows, model)
    with open(argv[2], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["dtype"] == "float64":
            print(
                f"{model:6} {row['quantity']:28} max|diff| {row['max_abs_diff']}  "
                f"rel to max {row['rel_to_max']}  {'ok' if row['pass'] else 'FAIL'}"
            )
    print(f"{model}: float64 gradients {'within' if ok else 'NOT within'} {TOL:g} of autograd")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
