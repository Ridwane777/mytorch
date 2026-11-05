#!/bin/sh
# Ensembles of our two networks against each alone (#40, ADR-0045, docs/performance/27-ensembles.md).
#   1. the split of ADR-0017 (seed 42), written by scripts/bench_baselines.sh (RUN);
#   2. on the validation part: each network alone, the weighted mean for several weights, the vote,
#      to choose the weights; the stacked combiner (configs/ensemble_stack.conf) is trained there;
#   3. on the test part, read once per model: each network alone, the plain mean, the mean with the
#      weights chosen in 2, the vote and the stack, through my_torch_analyzer --predict (time and
#      peak memory by GNU time), scored by scripts/score.cpp;
#   4. the table, the exact McNemar test of each ensemble against the CNN alone and the graph
#      (scripts/plot_ensembles.py).
# Usage: sh scripts/bench_ensembles.sh
# Environment: RUN (default runs/baselines), OUT (default docs/performance/data/ensembles),
# WEIGHTS (CNN weights tried, the dense one being 1; default "1 2 3 4 6"), PY (a Python with
# matplotlib, default bonus/python/.venv/bin/python).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/ensembles}"
WEIGHTS="${WEIGHTS:-1 2 3 4 6}"
PY="${PY:-bonus/python/.venv/bin/python}"
DENSE=my_torch_network_final.nn
CNN=bonus/networks/my_torch_network_cnn_epoch10.nn
WORK="$RUN/ensembles"

[ -s "$RUN/split/test.txt" ] || {
    echo "bench_ensembles.sh: $RUN/split/test.txt missing: run scripts/bench_baselines.sh" >&2
    exit 1
}
mkdir -p "$WORK/models" "$OUT"
make -s my_torch_analyzer build/libmy_torch.a
make -s -C bonus my_torch_ensemble
g++ -std=c++20 -O2 -pthread -Iinclude scripts/score.cpp build/libmy_torch.a -o "$WORK/score"
for part in validation test; do
    cut -d' ' -f1-6 "$RUN/split/$part.txt" > "$WORK/$part.boards"
done

# predict MODEL PART NAME: predictions, time and scores of MODEL on PART into $WORK/PART/NAME.
predict() {
    dir="$WORK/$2/$3"
    mkdir -p "$dir"
    /usr/bin/time -f "%e %M" -o "$dir/time.txt" ./my_torch_analyzer --predict "$1" \
        "$WORK/$2.boards" > "$dir/predictions.txt"
    "$WORK/score" "$DENSE" "$RUN/split/$2.txt" "$dir/predictions.txt" "$dir" > /dev/null
    stat -c %s "$1" > "$dir/bytes.txt"
}
accuracy() {
    sed -n 's/^accuracy: \([0-9.]*\).*/\1/p' "$1/scores.txt"
}

# 2. Choices on the validation part.
cp "$DENSE" "$WORK/models/dense.nn"
cp "$CNN" "$WORK/models/cnn.nn"
predict "$WORK/models/dense.nn" validation dense
predict "$WORK/models/cnn.nn" validation cnn
echo "cnn_weight,validation_accuracy" > "$OUT/weights.csv"
best=1
best_accuracy=0
for w in $WEIGHTS; do
    bonus/my_torch_ensemble average "$WORK/models/average_$w.nn" "$DENSE" "$CNN:$w" > /dev/null
    predict "$WORK/models/average_$w.nn" validation "average_$w"
    a=$(accuracy "$WORK/validation/average_$w")
    echo "$w,$a" >> "$OUT/weights.csv"
    if awk "BEGIN { exit !($a > $best_accuracy) }"; then
        best=$w
        best_accuracy=$a
    fi
done
echo "chosen CNN weight on validation: $best (accuracy $best_accuracy)" | tee "$OUT/choice.txt"
bonus/my_torch_ensemble vote "$WORK/models/vote.nn" "$DENSE" "$CNN" > /dev/null
predict "$WORK/models/vote.nn" validation vote
bonus/my_torch_ensemble stack "$WORK/models/stack.nn" configs/ensemble_stack.conf \
    "$RUN/split/validation.txt" "$DENSE" "$CNN" | tee "$OUT/stack_training.txt"

# 3. The test part, once per model.
cp "$WORK/models/average_$best.nn" "$WORK/models/weighted.nn"
for name in dense cnn average_1 weighted vote stack; do
    predict "$WORK/models/$name.nn" test "$name"
    mkdir -p "$OUT/test/$name"
    cp "$WORK/test/$name/scores.txt" "$WORK/test/$name/confusion.csv" \
        "$WORK/test/$name/time.txt" "$WORK/test/$name/bytes.txt" "$OUT/test/$name/"
done
{
    echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "commit: $(git rev-parse --short HEAD)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "threads: $(nproc)"
    echo "compiler: $(g++ --version | head -1)"
    echo "members: $DENSE, $CNN"
} > "$OUT/run.txt"
"$PY" scripts/plot_ensembles.py "$OUT" "$best" "$WORK/test" "$RUN/split/test.txt"
