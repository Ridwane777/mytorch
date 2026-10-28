#!/bin/sh
# Hyperparameter search of the chess network (#10, docs/performance/15-hyperparameters.md,
# ADR-0028, ADR-0029).
#   1. builds scripts/hyperparam_search.cpp against the library and the analyzer sources;
#   2. prepares the data like scripts/train_final.sh: split with SEED (ADR-0017), the first
#      FRACTION of the training and validation parts, the training part augmented (ADR-0019); the
#      test part is never used;
#   3. runs MODES in order (default "grid random seeds"): the 36-combination grid, a random search
#      of RANDOM_COUNT combinations (seed RANDOM_SEED) read from the grid's trials, then the TOP
#      best combinations again with SEEDS more training seeds;
#   4. draws the graphs with scripts/plot_hyperparameters.py.
# Every mode is resumable: rerunning the script skips the trials already in its log.
# Usage: sh scripts/hyperparam_search.sh [run directory (default runs/hyperparameters)]
# Environment: DATASET (default dataset), FRACTION (0.1), SEED (42), MODES, RANDOM_COUNT (12),
# RANDOM_SEED (7), TOP (3), SEEDS (2), TEMPLATE (configs/final_network.conf), OUT (graphs and CSV,
# default docs/performance/data/hyperparameters).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
RUN="${1:-runs/hyperparameters}"
DATASET="${DATASET:-dataset}"
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"
MODES="${MODES:-grid random seeds}"
RANDOM_COUNT="${RANDOM_COUNT:-12}"
RANDOM_SEED="${RANDOM_SEED:-7}"
TOP="${TOP:-3}"
SEEDS="${SEEDS:-2}"
TEMPLATE="${TEMPLATE:-configs/final_network.conf}"
OUT="${OUT:-docs/performance/data/hyperparameters}"
mkdir -p "$RUN" "$OUT"

# 1. Build.
make -s build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer \
    scripts/hyperparam_search.cpp $(ls src/analyzer/*.cpp | grep -v main.cpp) build/libmy_torch.a \
    -o "$RUN/hyperparam_search"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/augment.cpp \
    src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp -o "$RUN/augment"

# 2. Data, once: the same split and fraction as scripts/train_final.sh.
if [ ! -f "$RUN/train_augmented.txt" ]; then
    OUT="$RUN/split" SEED="$SEED" DATASET="$DATASET" sh scripts/dataset_split.sh \
        "$RUN/split_stats" > "$RUN/split.log"
    for part in train validation; do
        total=$(wc -l < "$RUN/split/$part.txt")
        keep=$(python3 -c "import math; print(max(1, math.floor($total * $FRACTION)))")
        head -n "$keep" "$RUN/split/$part.txt" > "$RUN/$part.txt"
    done
    "$RUN/augment" --all "$RUN/train.txt" "$RUN/train_augmented.txt"
fi

# 3. Search.
for mode in $MODES; do
    case "$mode" in
    grid) set -- grid ;;
    random) set -- random "$RANDOM_COUNT" "$RANDOM_SEED" ;;
    seeds) set -- seeds "$TOP" "$SEEDS" ;;
    *) echo "hyperparam_search.sh: unknown mode '$mode'" >&2; exit 1 ;;
    esac
    "$RUN/hyperparam_search" "$TEMPLATE" "$RUN/train_augmented.txt" "$RUN/validation.txt" "$RUN" "$@"
done

# 4. Results and graphs.
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads"
    echo "split seed: $SEED, fraction: $FRACTION"
    echo "training lines: $(wc -l < "$RUN/train_augmented.txt"), validation lines: $(wc -l < "$RUN/validation.txt")"
} > "$OUT/run.txt"
for f in grid.csv details.csv seeds.csv "random_$RANDOM_SEED.csv"; do
    [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/"
done
python3 scripts/plot_hyperparameters.py "$OUT" "$RANDOM_SEED"
