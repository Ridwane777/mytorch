#!/usr/bin/env python3
"""Exact McNemar test between two prediction files on the same labelled positions: positions B
fixes against A, positions B breaks, and the two-sided p-value.
Usage: mcnemar.py LABELLED_FILE PREDICTIONS_A PREDICTIONS_B"""
import sys
from math import comb


def main(labelled, path_a, path_b):
    truth = [" ".join(line.split()[6:]) for line in open(labelled)]
    a = [line.strip() for line in open(path_a)]
    b = [line.strip() for line in open(path_b)]
    if not len(truth) == len(a) == len(b):
        sys.exit("mcnemar.py: the files have different line counts")
    fixed = sum(1 for t, x, y in zip(truth, a, b) if x != t and y == t)
    broken = sum(1 for t, x, y in zip(truth, a, b) if x == t and y != t)
    n = fixed + broken
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, broken) + 1)) / 2 ** n) if n else 1.0
    print(f"positions: {len(truth)}\nfixed: {fixed}\nbroken: {broken}\np: {p:.2g}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
