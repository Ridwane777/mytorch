#!/bin/sh
# Normalization benchmark (#30, docs/performance/17-normalization.md): trains the six networks of
# configs/normalization/ (final architecture and a deep one, each without normalization, with
# BatchNorm and with LayerNorm) one after the other, with scripts/train_final.sh at 10 % of the
# data, scored on the validation split, then draws the learning curves with
# scripts/plot_normalization.py. Resumable: a run whose scores exist is not trained again.
# Usage: sh scripts/bench_normalization.sh
# Environment: DATASET (default dataset), FRACTION (default 0.1), OUT (default
# docs/performance/data/normalization).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
FRACTION="${FRACTION:-0.1}"
DEST="${OUT:-docs/performance/data/normalization}"
WORK=runs/normalization
mkdir -p "$WORK" "$DEST"

make -s
BUILD=1
for config in configs/normalization/shallow_none.conf configs/normalization/shallow_batch.conf \
    configs/normalization/shallow_layer.conf configs/normalization/deep_none.conf \
    configs/normalization/deep_batch.conf configs/normalization/deep_layer.conf; do
    name=$(basename "$config" .conf)
    if [ ! -s "$WORK/$name/scores.txt" ]; then
        echo "training $name"
        FRACTION="$FRACTION" EVAL=validation BUILD="$BUILD" \
            sh scripts/train_final.sh "$config" "$WORK/$name" > "$WORK/$name.log" 2>&1
    fi
    BUILD=0
    mkdir -p "$DEST/$name"
    cp "$WORK/$name/curve.csv" "$WORK/$name/scores.txt" "$WORK/$name/run.txt" "$DEST/$name/"
done
python3 scripts/plot_normalization.py "$DEST"
