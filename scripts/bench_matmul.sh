#!/bin/sh
# Measures the blocked, threaded matrix product (docs/performance/12-matmul.md, ADR-0024):
#   1. scripts/bench_matmul.cpp: speed of the three products of a 901 -> 256 layer, plain loop
#      against blocked on 1 to all threads;
#   2. one training run of the final network (configs/final_network.conf, 2 epochs, no early
#      stopping) on SAMPLE dataset positions, with MY_TORCH_THREADS from 1 to all threads; the
#      trained networks must be byte-identical;
#   3. scripts/plot_matmul.py draws the graphs.
# Usage: sh scripts/bench_matmul.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), SAMPLE (default 50000), THREADS (default "1 2 4 8 N",
# N = number of hardware threads).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
SAMPLE="${SAMPLE:-50000}"
N="$(nproc)"
THREADS="${THREADS:-1 2 4 8 $N}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT"

make -s my_torch_analyzer my_torch_generator
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/bench_matmul.cpp \
    src/my_torch/*.cpp -o "$WORK/bench_matmul"
# shellcheck disable=SC2086
"$WORK/bench_matmul" "$OUT" $THREADS

python3 - "$DATASET" "$SAMPLE" > "$WORK/sample.txt" <<'PYTHON'
import glob, random, sys
lines = [l.strip() + "\n" for p in sorted(glob.glob(sys.argv[1] + "/*/*.txt"))
         for l in open(p) if l.strip()]
sys.stdout.writelines(random.Random(3).sample(lines, int(sys.argv[2])))
PYTHON
sed -e 's/^train .*/train epochs=2 batch=64 seed=42 patience=0 validation_tail=0/' \
    configs/final_network.conf > "$WORK/net.conf"
(cd "$WORK" && "$ROOT/my_torch_generator" net.conf 1)
echo "threads,seconds,peak_kb" > "$OUT/matmul_training.csv"
for t in $THREADS; do
    cp "$WORK/net_1.nn" "$WORK/t$t.nn"
    MY_TORCH_THREADS=$t /usr/bin/time -f "%e %M" -o "$WORK/t$t.time" \
        ./my_torch_analyzer --train "$WORK/t$t.nn" "$WORK/sample.txt" > /dev/null
    read -r s kb < "$WORK/t$t.time"
    echo "$t,$s,$kb" >> "$OUT/matmul_training.csv"
done
same=yes
for t in $THREADS; do cmp -s "$WORK/t1.nn" "$WORK/t$t.nn" || same=no; done
echo "networks identical for every thread count: $same" > "$OUT/matmul_training.txt"
python3 scripts/plot_matmul.py "$OUT"
