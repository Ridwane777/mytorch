#!/bin/sh
# Reproduces the optimizer and scheduler comparison (docs/performance/optimiseurs.md).
# Builds the benchmark against src/my_torch, runs it, then draws the graphs (matplotlib).
set -eu

cd "$(dirname "$0")/.."
OUT="${1:-docs/performance/data}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$OUT"
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/bench_optimizers.cpp \
    src/my_torch/*.cpp -o "$WORK/bench_optimizers"
"$WORK/bench_optimizers" "$OUT"
python3 scripts/plot_optimizers.py "$OUT"
