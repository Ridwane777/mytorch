#!/bin/sh
# MC dropout measurement (#26, docs/performance/18-uncertainty.md): Monte Carlo predictions of a
# network with dropout on the test split (split seed 42, ADR-0017) by scripts/uncertainty.cpp, then
# the analysis and graphs by scripts/plot_uncertainty.py.
# The default network is the candidate of the hyperparameter search (dropout 0.2, 100 % of the
# data), written by: sh scripts/train_final.sh configs/candidate_network.conf (about 4 hours); the
# pushed network has no dropout, so every pass would be the same.
# Usage: sh scripts/uncertainty.sh
# Environment: NETWORK (default runs/candidate_network/network.nn), DATASET (default dataset),
# PASSES (default 30), SEED (default 1), OUT (default docs/performance/data/uncertainty).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
NETWORK="${NETWORK:-runs/candidate_network/network.nn}"
DATASET="${DATASET:-dataset}"
PASSES="${PASSES:-30}"
SEED="${SEED:-1}"
DEST="${OUT:-docs/performance/data/uncertainty}"
WORK=runs/uncertainty
mkdir -p "$WORK" "$DEST"
[ -f "$NETWORK" ] || {
    echo "uncertainty.sh: $NETWORK not found (see the header of this script)" >&2
    exit 1
}

make -s
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/uncertainty.cpp \
    src/analyzer/Cli.cpp src/analyzer/Encoding.cpp src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp \
    build/libmy_torch.a -o "$WORK/uncertainty"
OUT="$WORK/split" DATASET="$DATASET" sh scripts/dataset_split.sh "$WORK/split_stats" \
    > "$WORK/split.log"
start=$(date +%s)
"$WORK/uncertainty" "$NETWORK" "$WORK/split/test.txt" "$WORK/positions.csv" "$PASSES" "$SEED"
seconds=$(( $(date +%s) - start ))
python3 scripts/plot_uncertainty.py "$WORK/positions.csv" "$DEST" | tee "$DEST/errors.txt"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "network: $NETWORK, sha256 $(sha256sum "$NETWORK" | cut -d' ' -f1)"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(free -g | awk 'NR==2{print $2}') GB RAM, $(uname -sr)"
    echo "positions: $(($(wc -l < "$WORK/split/test.txt"))), passes: $PASSES, seed: $SEED"
    echo "plain and MC predictions: $seconds s"
} > "$DEST/run.txt"
cat "$DEST/run.txt"
