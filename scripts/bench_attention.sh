#!/bin/sh
# A mini transformer against the CNN and the dense network (#32, ADR-0047,
# docs/performance/29-attention.md). Same budget for the four networks of configs/attention/:
#   - training: the first FRACTION of the training part of the split (ADR-0017, seed 42), no
#     augmentation, followed by the whole validation part used as validation_tail (ADR-0022);
#     10 epochs, no early stopping, the same seeds;
#   - each network trained by my_torch_analyzer --train (time and peak memory by GNU time; one line
#     per epoch with the validation loss and accuracy), then scored ONCE on the test part
#     (scripts/score.cpp); the test part is never used to choose anything;
#   - every other network compared with the CNN on the same test positions (exact McNemar test,
#     scripts/mcnemar.py);
#   - the table and the graphs (scripts/plot_attention.py).
# Budget (FRACTION = 0.1, same for the four networks): one epoch of transformer1 at 0.1 took 251 s
# on this machine (load average 5 to 9, shared desktop; docs/performance/data/attention/pilot.txt),
# so 10 epochs take about 42 minutes, and two blocks about twice that: with the CNN and the dense
# network the whole script is expected to finish in about 2.5 hours. A larger fraction would not fit
# the 3 hour limit; dim 32 and 4 heads were kept, no pilot of other sizes was needed.
# Needs the split of ADR-0017: created here, with scripts/dataset_split.sh as scripts/bench_baselines.sh
# does, when RUN/split is missing.
# Usage: sh scripts/bench_attention.sh
# Environment: RUN (default runs/baselines), OUT (default docs/performance/data/attention),
# FRACTION (default 0.1), NETS (default "dense cnn transformer1 transformer2"), PY (default
# python3, with matplotlib), FORCE (1 to train again the networks already finished in RUN).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT=$(pwd)
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/attention}"
FRACTION="${FRACTION:-0.1}"
NETS="${NETS:-dense cnn transformer1 transformer2}"
PY="${PY:-python3}"
FORCE="${FORCE:-0}"
WORK="$RUN/attention"

if [ ! -s "$RUN/split/test.txt" ]; then
    mkdir -p "$RUN"
    OUT="$RUN/split" SEED=42 DATASET="${DATASET:-dataset}" sh scripts/dataset_split.sh \
        "$RUN/split_stats" > "$RUN/split.log"
fi
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
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "load average at the start: $(cut -d' ' -f1-3 /proc/loadavg)"

for name in $NETS; do
    dir="$WORK/$name"
    mkdir -p "$dir" "$OUT/$name"
    if [ "$FORCE" != 1 ] && [ -s "$dir/scores.txt" ]; then
        echo "$name: already trained in $dir (FORCE=1 to train again)"
    else
        sed "s/validation_tail=0/validation_tail=$valid/" "configs/attention/$name.conf" \
            > "$dir/$name.conf"
        (cd "$dir" && "$ROOT/my_torch_generator" "$name.conf" 1 > /dev/null)
        cut -d' ' -f1-3 /proc/loadavg > "$dir/loadavg_before.txt"
        /usr/bin/time -f "%e %M" -o "$dir/train_time.txt" ./my_torch_analyzer --train \
            --save "$dir/network.nn" "$dir/${name}_1.nn" "$WORK/train.txt" > "$dir/epochs.txt"
        cut -d' ' -f1-3 /proc/loadavg > "$dir/loadavg_after.txt"
        /usr/bin/time -f "%e %M" -o "$dir/predict_time.txt" ./my_torch_analyzer --predict \
            "$dir/network.nn" "$WORK/test.boards" > "$dir/predictions.txt"
        "$WORK/score" "$dir/network.nn" "$RUN/split/test.txt" "$dir/predictions.txt" "$dir" \
            > /dev/null
        stat -c %s "$dir/network.nn" > "$dir/bytes.txt"
    fi
    cp "$dir/epochs.txt" "$dir/train_time.txt" "$dir/predict_time.txt" "$dir/scores.txt" \
        "$dir/confusion.csv" "$dir/bytes.txt" "$dir/loadavg_before.txt" "$dir/loadavg_after.txt" \
        "$OUT/$name/"
    echo "$name: $(sed -n 's/^accuracy: \([0-9.]*\).*/\1/p' "$dir/scores.txt")"
done

# Each network against the CNN, same test positions (A = the CNN, B = the network).
if [ -s "$WORK/cnn/predictions.txt" ]; then
    for name in $NETS; do
        [ "$name" = cnn ] && continue
        python3 scripts/mcnemar.py "$RUN/split/test.txt" "$WORK/cnn/predictions.txt" \
            "$WORK/$name/predictions.txt" > "$OUT/mcnemar_$name.txt"
    done
fi
{
    echo "started: $started"
    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!runs' || echo ' (uncommitted changes)')"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(free -m | awk '/Mem:/ {print $2}') MB RAM, $(uname -sr)"
    echo "compiler: $(g++ --version | head -1)"
    echo "load average at the end: $(cut -d' ' -f1-3 /proc/loadavg) (1, 5, 15 minutes)"
    echo "fraction: $FRACTION of the training part ($keep positions), validation_tail $valid, 10 epochs"
    echo "networks: $NETS"
    echo "command: FRACTION=$FRACTION NETS=\"$NETS\" sh scripts/bench_attention.sh"
} > "$OUT/run.txt"
"$PY" scripts/plot_attention.py "$OUT" $NETS
