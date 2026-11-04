#!/bin/sh
# Classical baselines against our network (#43, ADR-0043, docs/performance/23-baselines.md).
# Needs bonus/comparaison/.venv with scikit-learn and LightGBM (bonus/comparaison/README.md): they live
# only there, never in the library or in what the Makefile builds (ADR-0036).
#   1. the split of ADR-0017 (seed 42), the same as the pushed network's;
#   2. every part encoded with fen-attacks-901 by `pytorch_reference export` (our own encoding);
#   3. each model family by bonus/comparaison/baselines.py under GNU time: its grid trained on the
#      whole training part (no augmentation) and chosen on the validation part, the chosen model
#      scored once on the test part;
#   4. the table and the graphs (scripts/plot_baselines.py).
# Usage: sh scripts/bench_baselines.sh
# Environment: DATASET (default dataset), SEED (split seed, 42), RUN (default runs/baselines, large
# files, git-ignored), OUT (default docs/performance/data/baselines), MODELS (default
# "logreg lightgbm forest mlp"), THREADS (default 0 = every hardware thread).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
DATASET="${DATASET:-dataset}"
SEED="${SEED:-42}"
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/baselines}"
MODELS="${MODELS:-logreg lightgbm forest mlp}"
THREADS="${THREADS:-0}"
PY=bonus/comparaison/.venv/bin/python
HELPER="$RUN/bin/pytorch_reference"

fail() {
    echo "bench_baselines.sh: $1" >&2
    exit 1
}

[ -x "$PY" ] || fail "$PY not found: create the virtualenv first (bonus/comparaison/README.md)"
mkdir -p "$RUN/bin" "$RUN/data" "$OUT"

# 1-2. Split and encoding.
make -s build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer \
    scripts/pytorch_reference.cpp $(ls src/analyzer/*.cpp | grep -v main.cpp) build/libmy_torch.a \
    -o "$HELPER"
if [ ! -s "$RUN/split/train.txt" ]; then
    OUT="$RUN/split" SEED="$SEED" DATASET="$DATASET" sh scripts/dataset_split.sh \
        "$RUN/split_stats" > "$RUN/split.log"
fi
for part in train validation test; do
    if [ ! -s "$RUN/data/$part.y.u8" ]; then
        "$HELPER" export configs/final_network.conf "$RUN/split/$part.txt" "$RUN/data/$part" \
            > "$RUN/data/$part.shape"
    fi
done

# 2b. Our networks on the same test part and the same machine, for the table: the pushed dense
# network and the epoch-10 CNN snapshot (their training times come from their own run.txt files).
make -s my_torch_analyzer
g++ -std=c++20 -O2 -pthread -Iinclude scripts/score.cpp build/libmy_torch.a -o "$RUN/bin/score"
cut -d' ' -f1-6 "$RUN/split/test.txt" > "$RUN/test_boards.txt"
for net in my_torch_network_final.nn bonus/networks/my_torch_network_cnn_epoch10.nn; do
    name=$(basename "$net" .nn)
    dir="$RUN/ours/$name"
    mkdir -p "$dir" "$OUT/ours/$name"
    /usr/bin/time -f "%e %M" -o "$dir/time.txt" ./my_torch_analyzer --predict "$net" \
        "$RUN/test_boards.txt" > "$dir/predictions.txt"
    "$RUN/bin/score" "$net" "$RUN/split/test.txt" "$dir/predictions.txt" "$dir" > /dev/null
    echo "$(stat -c %s "$net")" > "$dir/bytes.txt"
    cp "$dir/scores.txt" "$dir/time.txt" "$dir/bytes.txt" "$OUT/ours/$name/"
done

# 3. One process per family, so that GNU time gives each its own peak memory.
for model in $MODELS; do
    dir="$RUN/$model"
    mkdir -p "$dir"
    /usr/bin/time -f "%e %M" -o "$dir/time.txt" "$PY" -W ignore bonus/comparaison/baselines.py \
        --model "$model" --train "$RUN/data/train" --valid "$RUN/data/validation" \
        --test "$RUN/data/test" --threads "$THREADS" --seed "$SEED" --out-dir "$dir" \
        | tee "$dir/log.txt"
    mkdir -p "$OUT/$model"
    cp "$dir/grid.csv" "$dir/scores.txt" "$dir/confusion.csv" "$dir/summary.csv" "$dir/time.txt" \
        "$OUT/$model/"
done

# 4. Table and graphs, then how the numbers were made.
python3 scripts/plot_baselines.py "$OUT"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . || echo ' (uncommitted changes)')"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(free -m | awk '/Mem:/ {print $2}') MB RAM, $(uname -sr)"
    echo "python: $("$PY" --version 2>&1), $("$PY" -c 'import sklearn, lightgbm, numpy; print("scikit-learn", sklearn.__version__, "lightgbm", lightgbm.__version__, "numpy", numpy.__version__)')"
    echo "split seed: $SEED, rows: train $(wc -l < "$RUN/split/train.txt"), validation $(wc -l < "$RUN/split/validation.txt"), test $(wc -l < "$RUN/split/test.txt")"
    echo "threads: $THREADS (0 = every hardware thread)"
} > "$OUT/run.txt"
cat "$OUT/run.txt"
