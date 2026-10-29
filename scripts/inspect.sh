#!/bin/sh
# Network inspection (#36, docs/performance/19-inspection.md):
#   1. the summary of the pushed network;
#   2. weight and gradient histograms per epoch: the final configuration trained at 10 % of the
#      data (scripts/train_final.sh prepares the files and trains it with my_torch_analyzer), then
#      the same training again by scripts/inspect.cpp with an epoch callback that exports them;
#   3. the saliency of the squares of every test position in check or mate, for the pushed network;
#   4. the graphs, with scripts/plot_inspection.py.
# Usage: sh scripts/inspect.sh
# Environment: NETWORK (default my_torch_network_final.nn), DATASET (default dataset), OUT (default
# docs/performance/data/inspection).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
NETWORK="${NETWORK:-my_torch_network_final.nn}"
DATASET="${DATASET:-dataset}"
DEST="${OUT:-docs/performance/data/inspection}"
WORK=runs/inspection
mkdir -p "$WORK" "$DEST"

make -s
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer -Ibonus/oracle -Ibonus/gui \
    scripts/inspect.cpp bonus/gui/Saliency.cpp bonus/oracle/Oracle.cpp src/analyzer/Cli.cpp \
    src/analyzer/Encoding.cpp src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp build/libmy_torch.a \
    -o "$WORK/inspect"

# 1. Summary.
"$WORK/inspect" summary "$NETWORK" | tee "$DEST/summary.txt"

# 2. Histograms: the training files of a 10 % run, then the same training with the callback.
if [ ! -s "$WORK/train/scores.txt" ]; then
    FRACTION=0.1 EVAL=validation DATASET="$DATASET" BUILD=0 \
        sh scripts/train_final.sh configs/final_network.conf "$WORK/train" > "$WORK/train.log"
fi
"$WORK/inspect" histograms "$WORK/train/final_network_1.nn" "$WORK/train/training_file.txt" \
    "$WORK/histograms.csv" | tee "$DEST/histograms_training.log"
echo "my_torch_analyzer --train on the same files:" >> "$DEST/histograms_training.log"
cat "$WORK/train/training.log" >> "$DEST/histograms_training.log"

# 3. Saliency on the test split.
OUT="$WORK/split" DATASET="$DATASET" sh scripts/dataset_split.sh "$WORK/split_stats" \
    > "$WORK/split.log"
start=$(date +%s)
"$WORK/inspect" saliency "$NETWORK" "$WORK/split/test.txt" "$WORK/saliency.csv"
echo "saliency of $(($(wc -l < "$WORK/saliency.csv") - 1)) positions: $(( $(date +%s) - start )) s" \
    > "$DEST/run.txt"

# 4. Graphs.
python3 scripts/plot_inspection.py "$WORK/histograms.csv" "$WORK/saliency.csv" \
    "$WORK/split/test.txt" "$DEST" | tee -a "$DEST/run.txt"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes)')"
    echo "network: $NETWORK, sha256 $(sha256sum "$NETWORK" | cut -d' ' -f1)"
    echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(uname -sr)"
} >> "$DEST/run.txt"
cat "$DEST/run.txt"
