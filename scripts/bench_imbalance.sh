#!/bin/sh
# Compares four ways of handling class imbalance on the real dataset (docs/performance/04-imbalance.md,
# ADR-0018): none, loss weighting, oversampling and undersampling of the minority classes.
#   1. TRAIN + TEST positions drawn at random from the dataset with a fixed seed (Python's
#      random.sample): the last TEST form the test set, kept in its natural distribution and never
#      resampled; the first TRAIN are the training sample, the only data the options may change;
#   2. scripts/resample_imbalance.py derives from the training sample the file of each option: the
#      sample itself (none, weighted), its minority classes duplicated up to the majority count
#      (oversample) or each class drawn down to the minority count (undersample); for weighted it
#      also computes the inverse frequency weights, injected in a copy of the configuration;
#   3. for each configs/imbalance/*.conf, my_torch_generator writes 3 networks: it derives the
#      initial weights of each from `init seed` + its index, so the three differ and the same
#      command always gives the same ones (and the same ones for every option);
#   4. each network is trained with my_torch_analyzer --train on the file of its option, timed
#      (wall time, peak memory), then predicts the test set; scripts/score_imbalance.py appends one
#      row to imbalance_runs.csv;
#   5. scripts/plot_imbalance.py draws the graphs.
# Usage: sh scripts/bench_imbalance.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), TRAIN (default 20000), TEST (default 5000),
#   SEED (default 1: the draw of the sample and of the resampling), EPOCHS (default: the 20 of the
#   configurations; a smaller value is only meant for a quick check).
set -eu

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
TRAIN="${TRAIN:-20000}"
TEST="${TEST:-5000}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-}"
RUNS=3
OPTIONS="none weighted oversample undersample"

fail() {
    echo "bench_imbalance.sh: $1" >&2
    exit 1
}

set -- "$DATASET"/*/*.txt
[ -e "$1" ] || fail "$DATASET/ not found or empty: extract the dataset (dataset.zip, given by the school) at the root of the repository, or set DATASET to its directory"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

make -s
mkdir -p "$OUT/imbalance_logs"
OUT="$(cd "$OUT" && pwd)"
rm -f "$OUT/imbalance_runs.csv" "$OUT"/imbalance_logs/*.log

# A seeded draw without replacement (random.Random(SEED).sample): the same dataset always gives
# the same positions, in the same order, whatever the classes' order in the files.
cat "$DATASET"/*/*.txt | python3 -c 'import random, sys
lines = sys.stdin.readlines()
n = int(sys.argv[1])
sys.stdout.writelines(random.Random(int(sys.argv[2])).sample(lines, min(n, len(lines))))' \
    $((TRAIN + TEST)) "$SEED" > "$WORK/all.txt"
[ "$(wc -l < "$WORK/all.txt")" -eq $((TRAIN + TEST)) ] ||
    fail "the dataset holds fewer than $((TRAIN + TEST)) positions"
head -n "$TRAIN" "$WORK/all.txt" > "$WORK/train_natural.txt"
tail -n "$TEST" "$WORK/all.txt" > "$WORK/test_labeled.txt"
cut -d' ' -f1-6 "$WORK/test_labeled.txt" > "$WORK/test_boards.txt"
python3 scripts/resample_imbalance.py none "$SEED" "$WORK/test_labeled.txt" \
    2> "$OUT/imbalance_logs/test_counts.log" > /dev/null

for option in $OPTIONS; do
    # The configuration copy, with the number of epochs of EPOCHS if given.
    config="$WORK/$option.conf"
    if [ -n "$EPOCHS" ]; then
        sed "s/^\(train .*\)epochs=[0-9]*/\1epochs=$EPOCHS/" "configs/imbalance/$option.conf" > "$config"
    else
        cp "configs/imbalance/$option.conf" "$config"
    fi

    case "$option" in
    none | weighted) mode=none ;;
    oversample) mode=over ;;
    undersample) mode=under ;;
    esac
    python3 scripts/resample_imbalance.py "$mode" "$SEED" "$WORK/train_natural.txt" \
        > "$WORK/train_$option.txt" 2> "$OUT/imbalance_logs/${option}_resample.log"
    if [ "$option" = weighted ]; then
        weights="$(python3 scripts/resample_imbalance.py weights "$SEED" "$WORK/train_natural.txt" \
            "$config" 2>> "$OUT/imbalance_logs/weighted_resample.log")"
        sed "s/^\(loss cross_entropy\) weights=.*/\1 weights=$weights/" "$config" > "$config.tmp"
        mv "$config.tmp" "$config"
        grep -q "^loss cross_entropy weights=$weights\$" "$config" ||
            fail "could not inject the class weights in the weighted configuration"
        echo "weights: $weights" >> "$OUT/imbalance_logs/weighted_resample.log"
    fi
    train_size="$(wc -l < "$WORK/train_$option.txt")"

    (cd "$WORK" && "$ROOT/my_torch_generator" "$config" "$RUNS")
    run=1
    while [ "$run" -le "$RUNS" ]; do
        echo "$option, network $run of $RUNS" >&2
        /usr/bin/time -f "%e %M" -o "$WORK/time.txt" \
            ./my_torch_analyzer --train "$WORK/${option}_$run.nn" "$WORK/train_$option.txt" \
            > "$OUT/imbalance_logs/${option}_$run.log"
        ./my_torch_analyzer --predict "$WORK/${option}_$run.nn" "$WORK/test_boards.txt" \
            > "$WORK/predictions.txt"
        python3 scripts/score_imbalance.py "$OUT/imbalance_runs.csv" "$option" "$run" \
            "$train_size" "$WORK/test_labeled.txt" "$WORK/predictions.txt" "$WORK/time.txt"
        run=$((run + 1))
    done
done

python3 scripts/plot_imbalance.py "$OUT"
