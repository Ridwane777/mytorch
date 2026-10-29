#!/bin/sh
# Benchmarks and charts of issue #18 (docs/performance/16-benchmarks.md):
#   1. accuracy per category of position (piece giving check, pieces on the board, side in check)
#      of the pushed network on the test split, categories from the rule-based oracle;
#   2. prediction time of my_torch_analyzer --predict on 1,000, 10,000 and 71,555 positions;
#   3. cost of the non-finite checks (ADR-0031, against commit BASE, before them) and of a
#      checkpoint per epoch (ADR-0032), on the final architecture with synthetic data;
#   4. the graphs, with scripts/plot_benchmarks.py, including the learning curves of the trained
#      networks already in docs/performance/data.
# Run it on an idle machine: the times are only meaningful without other load.
# Usage: sh scripts/bench_report.sh
# Environment: DATASET (default dataset), NETWORK (default my_torch_network_final.nn), REPEATS
# (default 5), BASE (default 62c1c94), OUT (default docs/performance/data/benchmarks).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
DATASET="${DATASET:-dataset}"
NETWORK="${NETWORK:-my_torch_network_final.nn}"
REPEATS="${REPEATS:-5}"
BASE="${BASE:-62c1c94}"
DEST="${OUT:-docs/performance/data/benchmarks}"
WORK=runs/benchmarks
mkdir -p "$WORK" "$DEST"

make -s
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer -Ibonus/oracle \
    scripts/category_accuracy.cpp src/analyzer/Fen.cpp bonus/oracle/Oracle.cpp \
    -o "$WORK/category_accuracy"
git worktree remove --force "$WORK/base" > /dev/null 2>&1 || true
git worktree add -f --detach "$WORK/base" "$BASE" > /dev/null 2>&1
CXX_BENCH="g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror"
$CXX_BENCH -I"$WORK/base/include" scripts/bench_overheads.cpp "$WORK"/base/src/my_torch/*.cpp \
    -o "$WORK/bench_base"
$CXX_BENCH -DCHECKPOINT -Iinclude scripts/bench_overheads.cpp src/my_torch/*.cpp \
    -o "$WORK/bench_current"
git worktree remove --force "$WORK/base"

# 1. Accuracy per category on the test split (split seed 42, ADR-0017).
OUT="$WORK/split" DATASET="$DATASET" sh scripts/dataset_split.sh "$WORK/split_stats" \
    > "$WORK/split.log"
cut -d' ' -f1-6 "$WORK/split/test.txt" > "$WORK/test_boards.txt"
./my_torch_analyzer --predict "$NETWORK" "$WORK/test_boards.txt" > "$WORK/predictions.txt"
"$WORK/category_accuracy" "$WORK/split/test.txt" "$WORK/predictions.txt" \
    "$DEST/category_accuracy.csv"

load="$(cut -d' ' -f1-3 /proc/loadavg)"
# 2. Prediction time, REPEATS runs per size.
echo "positions,run,seconds" > "$DEST/predict_times.csv"
for size in 1000 10000 $(wc -l < "$WORK/test_boards.txt"); do
    head -n "$size" "$WORK/test_boards.txt" > "$WORK/boards_$size.txt"
    run=1
    while [ "$run" -le "$REPEATS" ]; do
        begin=$(date +%s.%N)
        ./my_torch_analyzer --predict "$NETWORK" "$WORK/boards_$size.txt" > /dev/null
        end=$(date +%s.%N)
        echo "$size,$run,$(python3 -c "print(round($end - $begin, 4))")" >> "$DEST/predict_times.csv"
        run=$((run + 1))
    done
done

# 3. Overheads, the three variants interleaved so that a change of load hits all of them.
echo "variant,run,seconds" > "$DEST/overheads.csv"
run=1
while [ "$run" -le "$REPEATS" ]; do
    rm -f "$WORK/bench.ckpt"
    echo "before ADR-0031,$run,$("$WORK/bench_base" 3)" >> "$DEST/overheads.csv"
    echo "non-finite checks,$run,$("$WORK/bench_current" 3)" >> "$DEST/overheads.csv"
    echo "checks and a checkpoint per epoch,$run,$("$WORK/bench_current" 3 "$WORK/bench.ckpt")" \
        >> "$DEST/overheads.csv"
    run=$((run + 1))
done
checkpoint_bytes=$(wc -c < "$WORK/bench.ckpt")

# 4. Run card and graphs.
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "network: $NETWORK, sha256 $(sha256sum "$NETWORK" | cut -d' ' -f1)"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(free -g | awk 'NR==2{print $2}') GB RAM, $(uname -sr)"
    echo "compiler: $(g++ --version | head -1)"
    echo "load average before the timings: $load"
    echo "repeats: $REPEATS, base commit for the overheads: $BASE"
    echo "checkpoint of the final architecture: $checkpoint_bytes bytes"
} > "$DEST/run.txt"
python3 scripts/plot_benchmarks.py "$DEST"
cat "$DEST/run.txt"
