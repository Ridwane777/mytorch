#!/bin/sh
# Compares training with and without symmetry augmentation (docs/performance/05-augmentation.md,
# ADR-0019).
#   1. TRAIN + TEST positions drawn at random from the dataset with a fixed seed (Python's
#      random.Random(SEED).sample): the first TRAIN go to train.txt, the last TEST to test.txt.
#      The split is done BEFORE any augmentation, so no symmetric twin of a test position is made
#      from a training one;
#   2. scripts/augment.cpp writes train_augmented.txt (4 x TRAIN positions) from train.txt ONLY;
#      test.txt is never augmented;
#   3. two arms use the same configuration (configs/augmentation.conf: fixed epochs, no
#      early stopping, validation 0 so that no symmetric twin sits in a validation split) and the
#      same initial weights: "base" trains on train.txt, "augmented" on train_augmented.txt. Each
#      network is timed, then predicts test.txt; scripts/score_augmentation.py appends one row to
#      augmentation_runs.csv;
#   4. scripts/plot_augmentation.py draws augmentation.png.
# Also written: augmentation_split.txt (sizes and the count of test positions found in the training
# files) and augmentation_logs/ (one training log per network).
# Usage: sh scripts/bench_augmentation.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), TRAIN (default 20000), TEST (default 5000), SEED
# (default 42, the draw), RUNS (default 3, networks per arm), CONFIG (default
# configs/augmentation.conf).
set -eu
# sort and comm must agree on one byte order whatever the user's locale.
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
TRAIN="${TRAIN:-20000}"
TEST="${TEST:-5000}"
SEED="${SEED:-42}"
RUNS="${RUNS:-3}"
CONFIG="${CONFIG:-configs/augmentation.conf}"

fail() {
    echo "bench_augmentation.sh: $1" >&2
    exit 1
}

set -- "$DATASET"/*/*.txt
[ -e "$1" ] || fail "$DATASET/ not found or empty: extract the dataset (dataset.zip, given by the school) at the root of the repository, or set DATASET to its directory"
[ -f "$CONFIG" ] || fail "configuration '$CONFIG' not found"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

make -s
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/augment.cpp \
    src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp -o "$WORK/augment"
mkdir -p "$OUT/augmentation_logs"
OUT="$(cd "$OUT" && pwd)"
rm -f "$OUT/augmentation_runs.csv" "$OUT"/augmentation_logs/*.log

# A seeded draw without replacement: the same dataset and SEED always give the same positions, in
# the same order. Blank lines are dropped, a missing final newline is added.
python3 - "$DATASET" $((TRAIN + TEST)) "$SEED" > "$WORK/all.txt" <<'PYTHON'
import glob
import random
import sys

lines = []
for path in sorted(glob.glob(sys.argv[1] + "/*/*.txt")):
    with open(path) as f:
        lines.extend(line.strip() + "\n" for line in f if line.strip())
sample = random.Random(int(sys.argv[3])).sample(lines, min(int(sys.argv[2]), len(lines)))
sys.stdout.writelines(sample)
PYTHON
[ "$(wc -l < "$WORK/all.txt")" -eq $((TRAIN + TEST)) ] ||
    fail "the dataset holds fewer than $((TRAIN + TEST)) positions"
head -n "$TRAIN" "$WORK/all.txt" > "$WORK/train.txt"
tail -n "$TEST" "$WORK/all.txt" > "$WORK/test.txt"
cut -d' ' -f1-6 "$WORK/test.txt" > "$WORK/test_boards.txt"
"$WORK/augment" --all "$WORK/train.txt" "$WORK/train_augmented.txt"

# Test positions that also appear in a training file (board, side, castling, en passant).
overlap() {
    cut -d' ' -f1-4 "$1" | sort -u > "$WORK/keys_train.txt"
    cut -d' ' -f1-4 "$WORK/test.txt" | sort -u > "$WORK/keys_test.txt"
    comm -12 "$WORK/keys_train.txt" "$WORK/keys_test.txt" | wc -l
}
{
    echo "dataset: $DATASET, seed $SEED"
    echo "train: $(wc -l < "$WORK/train.txt") positions"
    echo "train_augmented: $(wc -l < "$WORK/train_augmented.txt") positions"
    echo "test: $(wc -l < "$WORK/test.txt") positions"
    echo "distinct test positions found in train: $(overlap "$WORK/train.txt")"
    echo "distinct test positions found in train_augmented: $(overlap "$WORK/train_augmented.txt")"
} > "$OUT/augmentation_split.txt"
cat "$OUT/augmentation_split.txt" >&2

for arm in base augmented; do
    if [ "$arm" = base ]; then
        file="$WORK/train.txt"
    else
        file="$WORK/train_augmented.txt"
    fi
    positions="$(wc -l < "$file")"
    mkdir -p "$WORK/$arm"
    (cd "$WORK/$arm" && "$ROOT/my_torch_generator" "$ROOT/$CONFIG" "$RUNS")
    name="$(basename "$CONFIG" .conf)"
    run=1
    while [ "$run" -le "$RUNS" ]; do
        echo "$arm ($positions positions), network $run of $RUNS" >&2
        /usr/bin/time -f "%e %M" -o "$WORK/time.txt" \
            ./my_torch_analyzer --train "$WORK/$arm/${name}_$run.nn" "$file" \
            > "$OUT/augmentation_logs/${arm}_$run.log"
        ./my_torch_analyzer --predict "$WORK/$arm/${name}_$run.nn" "$WORK/test_boards.txt" \
            > "$WORK/predictions.txt"
        python3 scripts/score_augmentation.py "$OUT/augmentation_runs.csv" "$arm" "$positions" \
            "$run" "$WORK/test.txt" "$WORK/predictions.txt" "$WORK/time.txt"
        run=$((run + 1))
    done
done

python3 scripts/plot_augmentation.py "$OUT"
