#!/bin/sh
# Reproduces the network file measurement (docs/performance/13-serialization.md).
set -eu

cd "$(dirname "$0")/.."
OUT="${1:-docs/performance/data}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$OUT"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude scripts/bench_serialization.cpp \
    src/my_torch/*.cpp -o "$WORK/bench_serialization"
"$WORK/bench_serialization" "$OUT"
python3 scripts/plot_serialization.py "$OUT"
