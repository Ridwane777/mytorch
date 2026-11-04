#!/bin/sh
# Stratified k-fold cross-validation of the dense final network and of the CNN (#39,
# docs/performance/21-cross-validation.md, ADR-0039).
#   1. builds scripts/cross_validation.cpp against the library and the analyzer sources;
#   2. splits the dataset once with SEED (ADR-0017) and keeps the first DENSE_FRACTION and the
#      first CNN_FRACTION of the training part; the validation and test parts are never read;
#   3. cross-validates configs/final_network.conf ("dense") and configs/cnn_network.conf ("cnn"),
#      K folds each, a fixed number of epochs per fold, no early stopping, no augmentation;
#   4. writes run.txt and draws the graph and the summary with scripts/plot_cross_validation.py.
# The CNN costs far more per epoch than the dense network (about 158 s per epoch on 10 % of the
# data), hence its smaller fraction.
# Usage: sh scripts/bench_cross_validation.sh [run directory (default runs/cross_validation)]
# Environment: DATASET (default dataset), K (5), SEED (42), DENSE_FRACTION (0.05), DENSE_EPOCHS (5),
# CNN_FRACTION (0.01), CNN_EPOCHS (2), OUT (CSV and graph, default
# docs/performance/data/cross_validation).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
RUN="${1:-runs/cross_validation}"
DATASET="${DATASET:-dataset}"
K="${K:-5}"
SEED="${SEED:-42}"
DENSE_FRACTION="${DENSE_FRACTION:-0.05}"
DENSE_EPOCHS="${DENSE_EPOCHS:-5}"
CNN_FRACTION="${CNN_FRACTION:-0.01}"
CNN_EPOCHS="${CNN_EPOCHS:-2}"
OUT="${OUT:-docs/performance/data/cross_validation}"
mkdir -p "$RUN" "$OUT"

# 1. Build.
make -s build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer \
    scripts/cross_validation.cpp $(ls src/analyzer/*.cpp | grep -v main.cpp) build/libmy_torch.a \
    -o "$RUN/cross_validation"

# 2. Data: the training part of the split only.
OUT="$RUN/split" SEED="$SEED" DATASET="$DATASET" sh scripts/dataset_split.sh \
    "$RUN/split_stats" > "$RUN/split.log"
total=$(wc -l < "$RUN/split/train.txt")
subset() { # name, fraction: the first fraction of the training part, at least one line
    keep=$(python3 -c "import math; print(max(1, math.floor($total * $2)))")
    head -n "$keep" "$RUN/split/train.txt" > "$RUN/$1.txt"
}
subset dense "$DENSE_FRACTION"
subset cnn "$CNN_FRACTION"

# 3. Cross-validation, a fresh result file.
rm -f "$RUN/folds.csv"
wall() { # model: seconds of the run of that model
    start=$(date +%s)
    case "$1" in
    dense) "$RUN/cross_validation" configs/final_network.conf "$RUN/dense.txt" "$K" "$SEED" \
        "$DENSE_EPOCHS" "$RUN/folds.csv" dense ;;
    cnn) "$RUN/cross_validation" configs/cnn_network.conf "$RUN/cnn.txt" "$K" "$SEED" \
        "$CNN_EPOCHS" "$RUN/folds.csv" cnn ;;
    esac
    echo $(($(date +%s) - start))
}
DENSE_WALL=$(wall dense)
CNN_WALL=$(wall cnn)

# 4. Results and graph.
cp "$RUN/folds.csv" "$OUT/folds.csv"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads"
    echo "folds: $K, fold seed: $SEED, training split seed: $SEED"
    echo "training part of the split: $total lines (the validation and test parts are not read)"
    echo "dense: fraction $DENSE_FRACTION, $(wc -l < "$RUN/dense.txt") rows, $DENSE_EPOCHS epochs per fold, wall time $DENSE_WALL s"
    echo "cnn: fraction $CNN_FRACTION, $(wc -l < "$RUN/cnn.txt") rows, $CNN_EPOCHS epochs per fold, wall time $CNN_WALL s"
    echo "per fold: fresh network, no early stopping (patience 0), no augmentation"
} > "$OUT/run.txt"
python3 scripts/plot_cross_validation.py "$OUT"
