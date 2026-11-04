#!/bin/sh
# Text against binary network files (ADR-0042, docs/performance/24-binary-format.md):
#   1. size, save and load time of the pushed networks in both formats (scripts/bench_binary_format.cpp);
#   2. the predictions of the text and binary versions of each network on a test file, compared
#      line by line (they must be identical);
#   3. the graph (scripts/plot_binary_format.py).
# Usage: sh scripts/bench_binary_format.sh [TEST_FILE]   (default runs/final/split/test.txt, the
# test split of scripts/dataset_split.sh; without it, step 2 is skipped)
# Environment: OUT (default docs/performance/data/binary_format).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
TEST="${1:-runs/final/split/test.txt}"
OUT="${OUT:-docs/performance/data/binary_format}"
WORK=runs/binary_format
NETWORKS="my_torch_network_final.nn bonus/networks/my_torch_network_cnn_epoch10.nn"
mkdir -p "$WORK" "$OUT"

make -s
make -s -C bonus my_torch_nn_dump
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/bench_binary_format.cpp \
    build/libmy_torch.a -o "$WORK/bench_binary_format"

# 1. Size and timings.
"$WORK/bench_binary_format" measure "$OUT/binary_format.csv" $NETWORKS
cat "$OUT/binary_format.csv"

# 2. Same predictions from both formats.
: > "$OUT/equivalence.txt"
if [ -s "$TEST" ]; then
    cut -d' ' -f1-6 "$TEST" > "$WORK/boards.txt"
    for net in $NETWORKS; do
        name=$(basename "$net" .nn)
        bonus/my_torch_nn_dump "$net" "$WORK/$name.text.nn"
        "$WORK/bench_binary_format" convert "$net" "$WORK/$name.binary.nn"
        ./my_torch_analyzer --predict "$WORK/$name.text.nn" "$WORK/boards.txt" > "$WORK/$name.text.out"
        ./my_torch_analyzer --predict "$WORK/$name.binary.nn" "$WORK/boards.txt" \
            > "$WORK/$name.binary.out"
        if cmp -s "$WORK/$name.text.out" "$WORK/$name.binary.out"; then
            verdict="identical"
        else
            verdict="DIFFERENT"
        fi
        echo "$name: $(wc -l < "$WORK/boards.txt") positions, predictions $verdict" \
            | tee -a "$OUT/equivalence.txt"
    done
fi

# 3. Graph, then how the numbers were made.
python3 scripts/plot_binary_format.py "$OUT/binary_format.csv" "$OUT/binary_format.png"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' (uncommitted changes)')"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(uname -sr)"
    echo "compiler: $(g++ --version | head -n 1)"
    echo "timings: median of 9 runs, file on the local disk (page cache warm after the first run)"
} > "$OUT/run.txt"
cat "$OUT/run.txt"
