#!/bin/sh
# Compares the four input encodings on the real dataset (docs/performance/03-encodings.md, ADR-0016).
#   1. TRAIN + TEST positions drawn at random from the dataset with a fixed seed (Python's
#      random.sample): the first TRAIN to train (the network file holds out 10 % of them for early
#      stopping), the last TEST to score, never trained on;
#   2. for each configs/encodings/*.conf, my_torch_generator writes 3 networks: it derives the
#      initial weights of each from `init seed` + its index, so the three differ and the same
#      command always gives the same ones; the shuffling seed (`train seed`) is the same for all;
#   3. each network is trained with my_torch_analyzer --train, timed (wall time, peak memory), then
#      predicts the test set; scripts/score_encodings.py appends one row to encodings_runs.csv;
#   4. scripts/plot_encodings.py draws the graphs.
# Usage: sh scripts/bench_encodings.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), TRAIN (default 20000), TEST (default 5000).
set -eu

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
TRAIN="${TRAIN:-20000}"
TEST="${TEST:-5000}"
RUNS=3
ENCODINGS="onehot768 onehot773 signed385 attacks901"

fail() {
    echo "bench_encodings.sh: $1" >&2
    exit 1
}

set -- "$DATASET"/*/*.txt
[ -e "$1" ] || fail "$DATASET/ not found or empty: extract the dataset (dataset.zip, given by the school) at the root of the repository, or set DATASET to its directory"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

make -s
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/score.cpp build/libmy_torch.a \
    -o "$WORK/score"
mkdir -p "$OUT/encodings_logs"
OUT="$(cd "$OUT" && pwd)"
rm -f "$OUT/encodings_runs.csv" "$OUT"/encodings_logs/*.log

# A seeded draw without replacement (random.Random(1).sample): the same dataset always gives the
# same positions, in the same order, whatever the classes' order in the files.
cat "$DATASET"/*/*.txt | python3 -c 'import random, sys
lines = sys.stdin.readlines()
n = int(sys.argv[1])
sys.stdout.writelines(random.Random(1).sample(lines, min(n, len(lines))))' $((TRAIN + TEST)) \
    > "$WORK/all.txt"
[ "$(wc -l < "$WORK/all.txt")" -eq $((TRAIN + TEST)) ] ||
    fail "the dataset holds fewer than $((TRAIN + TEST)) positions"
head -n "$TRAIN" "$WORK/all.txt" > "$WORK/train.txt"
tail -n "$TEST" "$WORK/all.txt" > "$WORK/test_labeled.txt"
cut -d' ' -f1-6 "$WORK/test_labeled.txt" > "$WORK/test_boards.txt"

for name in $ENCODINGS; do
    config="$ROOT/configs/encodings/$name.conf"
    encoding="$(sed -n 's/^input \([^ ]*\) .*/\1/p' "$config")"
    (cd "$WORK" && "$ROOT/my_torch_generator" "$config" "$RUNS")
    run=1
    while [ "$run" -le "$RUNS" ]; do
        echo "$encoding, network $run of $RUNS" >&2
        /usr/bin/time -f "%e %M" -o "$WORK/time.txt" \
            ./my_torch_analyzer --train "$WORK/${name}_$run.nn" "$WORK/train.txt" \
            > "$OUT/encodings_logs/${name}_$run.log"
        ./my_torch_analyzer --predict "$WORK/${name}_$run.nn" "$WORK/test_boards.txt" \
            > "$WORK/predictions.txt"
        "$WORK/score" "$WORK/${name}_$run.nn" "$WORK/test_labeled.txt" "$WORK/predictions.txt" \
            "$WORK"
        python3 scripts/score_encodings.py "$OUT/encodings_runs.csv" "$encoding" "$run" \
            "$WORK/report.json" "$WORK/time.txt" \
            "$OUT/encodings_logs/${name}_$run.log"
        run=$((run + 1))
    done
done

python3 scripts/plot_encodings.py "$OUT"
