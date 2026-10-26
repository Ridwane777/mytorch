#!/bin/sh
# Dataset statistics and train/validation/test split (docs/performance/dataset.md, ADR-0017).
#   1. scripts/dataset_split.py reads the dataset, writes the statistics CSV files and a
#      dedupe-aware split stratified by label (fixed seed) to build/dataset_split/;
#   2. scripts/plot_dataset.py draws the charts from the CSV files.
# Usage: sh scripts/dataset_split.sh [statistics directory (default docs/performance/data)]
# Environment: DATASET (default dataset), OUT (default build/dataset_split), SEED (default 42).
set -eu

cd "$(dirname "$0")/.."
STATS="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
OUT="${OUT:-build/dataset_split}"
SEED="${SEED:-42}"

fail() {
    echo "dataset_split.sh: $1" >&2
    exit 1
}

set -- "$DATASET"/*/*.txt
[ -e "$1" ] || fail "$DATASET/ not found or empty: extract the dataset (dataset.zip, given by the school) at the root of the repository, or set DATASET to its directory"

python3 -I scripts/dataset_split.py --dataset "$DATASET" --out "$OUT" --stats "$STATS" --seed "$SEED"
# No -I here: it would hide a matplotlib installed in the user's site-packages.
python3 scripts/plot_dataset.py "$STATS"
