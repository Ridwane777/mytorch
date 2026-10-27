#!/bin/sh
# Measures the memory of my_torch_analyzer --train before and after training from a data source
# (docs/performance/11-training-memory.md, ADR-0021).
#   1. The "before" analyzer is built from BEFORE (default d2aae89, the last main commit that
#      encoded the whole training file at once), the "after" one from the working tree.
#   2. Same network file, same sample of SAMPLE dataset positions, one epoch each: peak memory,
#      time, and whether the two trained networks are byte-identical.
#   3. The "after" analyzer alone on the whole dataset with fen-attacks-901, one epoch: the memory
#      and the epoch time of the final configuration. The "before" one is not run there: it would
#      need more memory than the machine has (estimate written in the doc).
# Usage: sh scripts/bench_training_memory.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), SAMPLE (default 100000), BEFORE (default d2aae89).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
SAMPLE="${SAMPLE:-100000}"
BEFORE="${BEFORE:-d2aae89}"
WORK="$(mktemp -d)"
cleanup() {
    git -C "$ROOT" worktree remove --force "$WORK/before" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT

set -- "$DATASET"/*/*.txt
[ -e "$1" ] || { echo "bench_training_memory.sh: $DATASET/ not found" >&2; exit 1; }

git worktree add -q --detach "$WORK/before" "$BEFORE"
make -s -C "$WORK/before" my_torch_analyzer
make -s my_torch_analyzer my_torch_generator

# A sample of the dataset drawn with a fixed seed, and the whole dataset.
python3 - "$DATASET" "$SAMPLE" > "$WORK/sample.txt" <<'PYTHON'
import glob, random, sys
lines = [l.strip() + "\n" for p in sorted(glob.glob(sys.argv[1] + "/*/*.txt"))
         for l in open(p) if l.strip()]
sys.stdout.writelines(random.Random(1).sample(lines, int(sys.argv[2])))
PYTHON
cat "$DATASET"/*/*.txt > "$WORK/full.txt"

generate() { # encoding width: writes $WORK/<encoding>_1.nn, one hidden layer of 128
    cat > "$WORK/$1.conf" <<CONF
input $1 $2
labels 5
Nothing
Check White
Check Black
Checkmate White
Checkmate Black
optimizer adam lr=0.001
train epochs=1 batch=64 seed=42 validation=0.1
layers 4
dense $2 128
relu
dense 128 5
softmax
CONF
    (cd "$WORK" && "$ROOT/my_torch_generator" "$WORK/$1.conf" 1)
}
measure() { # run analyzer network data: one epoch, peak memory and time in the CSV
    cp "$WORK/$3" "$WORK/$1.nn"
    /usr/bin/time -f "%e %M" -o "$WORK/$1.time" "$2" --train "$WORK/$1.nn" "$4" > "$WORK/$1.log"
    read -r seconds kilobytes < "$WORK/$1.time"
    echo "$1,$(wc -l < "$4"),$seconds,$kilobytes" >> "$OUT/training_memory.csv"
}

mkdir -p "$OUT"
echo "run,positions,seconds,peak_kb" > "$OUT/training_memory.csv"
generate fen-onehot-768 768
measure before_768 "$WORK/before/my_torch_analyzer" fen-onehot-768_1.nn "$WORK/sample.txt"
measure after_768 "$ROOT/my_torch_analyzer" fen-onehot-768_1.nn "$WORK/sample.txt"
if cmp -s "$WORK/before_768.nn" "$WORK/after_768.nn"; then same=yes; else same=no; fi
echo "trained networks identical before/after ($SAMPLE positions): $same" \
    > "$OUT/training_memory.txt"
generate fen-attacks-901 901
measure after_901_full "$ROOT/my_torch_analyzer" fen-attacks-901_1.nn "$WORK/full.txt"
cat "$WORK/after_901_full.log" >> "$OUT/training_memory.txt"
python3 scripts/plot_training_memory.py "$OUT"
