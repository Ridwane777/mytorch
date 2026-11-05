#!/bin/sh
# Residual blocks against the same depth without shortcuts and the current CNN (#31, ADR-0046,
# docs/performance/28-residual.md). Same budget for the three networks of configs/residual/:
#   - training: the first FRACTION of the training part of the split (ADR-0017, seed 42), no
#     augmentation, followed by the whole validation part used as validation_tail (ADR-0022);
#     10 epochs, no early stopping, the same seeds;
#   - each network trained by my_torch_analyzer --train (time and peak memory by GNU time; one line
#     per epoch with the validation loss and accuracy), then scored once on the test part
#     (scripts/score.cpp);
#   - the table and the graph (scripts/plot_residual.py).
# Needs the split written by scripts/bench_baselines.sh (RUN).
# Usage: sh scripts/bench_residual.sh
# Environment: RUN (default runs/baselines), OUT (default docs/performance/data/residual),
# FRACTION (default 0.1), NETS (default "cnn cnn_deep cnn_residual"), PY (a Python with matplotlib,
# default bonus/python/.venv/bin/python).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT=$(pwd)
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/residual}"
FRACTION="${FRACTION:-0.1}"
NETS="${NETS:-cnn cnn_deep cnn_residual}"
PY="${PY:-bonus/python/.venv/bin/python}"
WORK="$RUN/residual"

[ -s "$RUN/split/test.txt" ] || {
    echo "bench_residual.sh: $RUN/split/test.txt missing: run scripts/bench_baselines.sh" >&2
    exit 1
}
mkdir -p "$WORK" "$OUT"
make -s my_torch_analyzer my_torch_generator build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Iinclude scripts/score.cpp build/libmy_torch.a -o "$WORK/score"

total=$(wc -l < "$RUN/split/train.txt")
keep=$(awk "BEGIN { printf \"%d\", $total * $FRACTION }")
valid=$(wc -l < "$RUN/split/validation.txt")
head -n "$keep" "$RUN/split/train.txt" > "$WORK/train.txt"
cat "$RUN/split/validation.txt" >> "$WORK/train.txt"
cut -d' ' -f1-6 "$RUN/split/test.txt" > "$WORK/test.boards"
echo "training positions: $keep, validation positions: $valid" | tee "$OUT/data.txt"

for name in $NETS; do
    dir="$WORK/$name"
    mkdir -p "$dir" "$OUT/$name"
    sed "s/validation_tail=0/validation_tail=$valid/" "configs/residual/$name.conf" > "$dir/$name.conf"
    (cd "$dir" && "$ROOT/my_torch_generator" "$name.conf" 1 > /dev/null)
    /usr/bin/time -f "%e %M" -o "$dir/train_time.txt" ./my_torch_analyzer --train \
        --save "$dir/network.nn" "$dir/${name}_1.nn" "$WORK/train.txt" > "$dir/epochs.txt"
    /usr/bin/time -f "%e %M" -o "$dir/predict_time.txt" ./my_torch_analyzer --predict \
        "$dir/network.nn" "$WORK/test.boards" > "$dir/predictions.txt"
    "$WORK/score" "$dir/network.nn" "$RUN/split/test.txt" "$dir/predictions.txt" "$dir" > /dev/null
    stat -c %s "$dir/network.nn" > "$dir/bytes.txt"
    cp "$dir/epochs.txt" "$dir/train_time.txt" "$dir/predict_time.txt" "$dir/scores.txt" \
        "$dir/confusion.csv" "$dir/bytes.txt" "$OUT/$name/"
    echo "$name: $(sed -n 's/^accuracy: \([0-9.]*\).*/\1/p' "$dir/scores.txt")"
done
# Residual network against the control without shortcuts, on the same test positions.
if [ -s "$WORK/cnn_deep/predictions.txt" ] && [ -s "$WORK/cnn_residual/predictions.txt" ]; then
    python3 scripts/mcnemar.py "$RUN/split/test.txt" "$WORK/cnn_deep/predictions.txt" \
        "$WORK/cnn_residual/predictions.txt" > "$OUT/mcnemar.txt"
fi
{
    echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "commit: $(git rev-parse --short HEAD)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "threads: $(nproc)"
    echo "compiler: $(g++ --version | head -1)"
} > "$OUT/run.txt"
"$PY" scripts/plot_residual.py "$OUT" $NETS
