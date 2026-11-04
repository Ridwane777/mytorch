#!/usr/bin/env python3
"""Python side of the bindings benchmark (ADR-0044, docs/performance/25-python-bindings.md).

Times Network.predict through the module on the same encoded rows as scripts/bench_python_bindings
(the C++ side), including the NumPy copy in and out, then checks that the probabilities are the
same bits as the C++ ones and that the argmax labels are those printed by my_torch_analyzer.
Usage: bench.py NETWORK X.u8 ROWS REPEATS CPP.f64 ANALYZER_PREDICTIONS CPP_TIMES OUT.csv"""
import csv
import sys
import time

import numpy as np

import my_torch


def main(network, x_path, rows, repeats, cpp_path, analyzer_path, cpp_times, out):
    net = my_torch.Network.load(network)
    rows, repeats = int(rows), int(repeats)
    inputs = np.fromfile(x_path, dtype=np.uint8, count=rows * net.inputs)
    inputs = inputs.reshape(rows, net.inputs).astype(np.float64)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        probabilities = net.predict(inputs)
        times.append(time.perf_counter() - start)

    cpp = np.fromfile(cpp_path, dtype=np.float64).reshape(probabilities.shape)
    identical = bool(np.array_equal(cpp, probabilities))
    max_diff = float(np.max(np.abs(cpp - probabilities)))
    labels = [net.labels[k] for k in probabilities.argmax(axis=1)]
    analyzer = open(analyzer_path).read().splitlines()[:rows]
    agree = sum(a == b for a, b in zip(labels, analyzer))
    cpp_seconds = [float(t) for t in open(cpp_times).read().split()]

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "repeat", "seconds"])
        for k, t in enumerate(cpp_seconds):
            writer.writerow(["cpp", k + 1, f"{t:.6f}"])
        for k, t in enumerate(times):
            writer.writerow(["python", k + 1, f"{t:.6f}"])
    print(f"rows: {rows}")
    print(f"cpp median: {np.median(cpp_seconds):.4f} s")
    print(f"python median: {np.median(times):.4f} s")
    print(f"identical probabilities: {identical} (max difference {max_diff:.3g})")
    print(f"labels equal to my_torch_analyzer: {agree}/{rows}")
    return 0 if identical and agree == rows else 1


if __name__ == "__main__":
    if len(sys.argv) != 9:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
