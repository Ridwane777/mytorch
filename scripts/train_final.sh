#!/bin/sh
# Trains a network from scratch, reproducibly (#17, ADR-0022), and scores it on the test split.
#   1. builds my_torch_analyzer, my_torch_generator and the augmentation tool;
#   2. splits the dataset with scripts/dataset_split.sh (ADR-0017): train, validation, test;
#   3. with FRACTION < 1, keeps the first FRACTION of each split (they are already shuffled);
#   4. augments the training split only (scripts/augment.cpp --all, ADR-0019), unless AUGMENT=0;
#   5. writes the training file: augmented training lines, then the validation lines, untouched,
#      at the end; the configuration's validation_tail is set to their count (ADR-0022);
#   6. generates the network from CONFIG, trains it, timed;
#   7. predicts the test split (or, with EVAL=validation, the validation split, so that trial runs
#      choose without ever looking at the test split), scores it with scripts/score.cpp (the
#      library's metrics, ADR-0026) and draws the graphs with scripts/plot_run.py;
#   8. writes run.txt: commit, configuration checksum, machine, compiler, sizes, times, memory.
# Usage: sh scripts/train_final.sh CONFIG [RUN_DIRECTORY (default runs/<config name>)]
# Environment: DATASET (default dataset), FRACTION (default 1), AUGMENT (default 1),
# SEED (split seed, default 42), EVAL (test or validation, default test), BUILD (default 1; 0
# skips the make step when the executables are already built, e.g. for parallel runs),
# BATCH (overrides the configuration's batch: a larger batch gives the threads more rows per
# step, at the price of fewer updates per epoch, so the result is no longer comparable with a
# run of the same configuration at its own batch size; run.txt records the train line),
# EPOCHS and PATIENCE (override the configuration's epochs and patience; PATIENCE=0 turns early
# stopping off, and then there is no restore of the best weights: network.nn holds the last
# epoch and the snapshots are the safety net), PROGRESS_EVERY (default 5: every N epochs, a
# progress line on stderr and the weights in network.nn.<date>-<time>.epochN.nn). The training
# state is saved in checkpoint.ckpt after every epoch: running the script again with the same
# RUN resumes it. A copy of the state of every epoch is kept in RUN/epochs/ (watch_epochs.sh).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
[ $# -ge 1 ] || { echo "usage: sh scripts/train_final.sh CONFIG [RUN_DIRECTORY]" >&2; exit 1; }
CONFIG="$1"
NAME="$(basename "$CONFIG" .conf)"
RUN="${2:-runs/$NAME}"
DATASET="${DATASET:-dataset}"
FRACTION="${FRACTION:-1}"
AUGMENT="${AUGMENT:-1}"
SEED="${SEED:-42}"
EVAL="${EVAL:-test}"
BUILD="${BUILD:-1}"
EPOCHS="${EPOCHS:-}"
PATIENCE="${PATIENCE:-}"
BATCH="${BATCH:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
fail() {
    echo "train_final.sh: $1" >&2
    exit 1
}
[ -f "$CONFIG" ] || fail "$CONFIG not found"
case "$EPOCHS$PROGRESS_EVERY$PATIENCE$BATCH" in
*[!0-9]*) fail "EPOCHS, PROGRESS_EVERY, PATIENCE and BATCH must be non-negative integers" ;;
esac
[ "$EVAL" = test ] || [ "$EVAL" = validation ] || fail "EVAL must be test or validation"
grep -q 'validation_tail=' "$CONFIG" || fail "$CONFIG must have validation_tail= in its train line"
set -- "$DATASET"/*/*.txt
[ -e "$1" ] || fail "$DATASET/ not found or empty"
mkdir -p "$RUN"
start=$(date +%s)

# 1. Build.
[ "$BUILD" = 0 ] || make -s my_torch_analyzer my_torch_generator
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/augment.cpp \
    src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp -o "$RUN/augment"
[ "$BUILD" = 0 ] || make -s build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/score.cpp build/libmy_torch.a \
    -o "$RUN/score"

# 2-3. Split, then keep a fraction of each part.
OUT="$RUN/split" SEED="$SEED" DATASET="$DATASET" sh scripts/dataset_split.sh "$RUN/split_stats" \
    > "$RUN/split.log"
for part in train validation test; do
    total=$(wc -l < "$RUN/split/$part.txt")
    keep=$(python3 -c "import math; print(max(1, math.floor($total * $FRACTION)))")
    head -n "$keep" "$RUN/split/$part.txt" > "$RUN/$part.txt"
done

# 4-5. Augment the training part only, then append the validation part untouched.
if [ "$AUGMENT" = 1 ]; then
    "$RUN/augment" --all "$RUN/train.txt" "$RUN/train_augmented.txt"
else
    cp "$RUN/train.txt" "$RUN/train_augmented.txt"
fi
tail_lines=$(wc -l < "$RUN/validation.txt")
cat "$RUN/train_augmented.txt" "$RUN/validation.txt" > "$RUN/training_file.txt"
sed "s/validation_tail=[0-9]*/validation_tail=$tail_lines/" "$CONFIG" > "$RUN/$NAME.conf"
# The key is anchored on a preceding blank: "max_epochs=" or any other key ending in "epochs" is
# left alone.
if [ -n "$BATCH" ]; then
    [ "$BATCH" -gt 0 ] || fail "BATCH must be positive"
    grep -q '[[:space:]]batch=' "$RUN/$NAME.conf" || fail "$CONFIG has no batch= to override"
    sed -i "s/\([[:space:]]\)batch=[0-9]*/\1batch=$BATCH/" "$RUN/$NAME.conf"
fi
[ -z "$EPOCHS" ] || sed -i "s/\([[:space:]]\)epochs=[0-9]*/\1epochs=$EPOCHS/" "$RUN/$NAME.conf"
if [ -n "$PATIENCE" ]; then
    grep -q '[[:space:]]patience=' "$RUN/$NAME.conf" || fail "$CONFIG has no patience= to override"
    sed -i "s/\([[:space:]]\)patience=[0-9]*/\1patience=$PATIENCE/" "$RUN/$NAME.conf"
fi

# 6. Generate and train. A copy of the state of every epoch is kept in $RUN/epochs/ meanwhile
# (scripts/watch_epochs.sh), and once more when the training ends, whatever the way.
(cd "$RUN" && "$ROOT/my_torch_generator" "$NAME.conf" 1)
sh scripts/watch_epochs.sh "$RUN" 20 > /dev/null 2>&1 &
WATCHER=$!
stop_watcher() {
    kill "$WATCHER" 2> /dev/null || true
    sh scripts/watch_epochs.sh "$RUN" 1 once > /dev/null 2>&1 || true
}
trap stop_watcher EXIT
# dash does not run the EXIT trap when a signal kills the script: turn the signals into an exit.
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
/usr/bin/time -f "%e %M" -o "$RUN/train_time.txt" \
    ./my_torch_analyzer --train --save "$RUN/network.nn" --checkpoint "$RUN/checkpoint.ckpt" \
    --progress-every "$PROGRESS_EVERY" "$RUN/${NAME}_1.nn" "$RUN/training_file.txt" \
    > "$RUN/training.log" 2>> "$RUN/progress.log"

# 7. Predict the evaluated split (test, or validation for trial runs) and score it.
cut -d' ' -f1-6 "$RUN/$EVAL.txt" > "$RUN/${EVAL}_boards.txt"
/usr/bin/time -f "%e %M" -o "$RUN/predict_time.txt" \
    ./my_torch_analyzer --predict "$RUN/network.nn" "$RUN/${EVAL}_boards.txt" \
    > "$RUN/predictions.txt"
"$RUN/score" "$RUN/network.nn" "$RUN/$EVAL.txt" "$RUN/predictions.txt" "$RUN"
python3 scripts/plot_run.py "$RUN/training.log" "$RUN"

# 8. Run card.
read -r train_seconds train_kb < "$RUN/train_time.txt"
read -r predict_seconds predict_kb < "$RUN/predict_time.txt"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "configuration: $CONFIG, sha256 $(sha256sum "$CONFIG" | cut -d' ' -f1)"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(free -g | awk 'NR==2{print $2}') GB RAM, $(uname -sr)"
    echo "compiler: $(g++ --version | head -1)"
    echo "train line: $(grep -m1 '^train ' "$RUN/$NAME.conf")"
    echo "split seed: $SEED, fraction: $FRACTION, augmentation: $AUGMENT, scored on: $EVAL"
    echo "training lines: $(wc -l < "$RUN/train_augmented.txt"), validation lines: $tail_lines, test lines: $(wc -l < "$RUN/test.txt")"
    echo "epochs run: $(grep -c '^epoch' "$RUN/training.log")"
    echo "training: $train_seconds s, peak $((train_kb / 1024)) MB"
    echo "prediction of the $EVAL split: $predict_seconds s, peak $((predict_kb / 1024)) MB"
    echo "total: $(( $(date +%s) - start )) s"
} > "$RUN/run.txt"
cat "$RUN/run.txt" "$RUN/scores.txt"
