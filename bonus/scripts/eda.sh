#!/bin/sh
# Exploratory analysis of the dataset (docs/performance/02-eda.md): builds bonus/my_torch_eda, runs it
# on every position (it uses the oracle to find the checking pieces), then draws the graphs.
# Usage: sh bonus/scripts/eda.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset).
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-docs/performance/data}"
DATASET="${DATASET:-dataset}"
make -s -C bonus my_torch_eda
./bonus/my_torch_eda "$DATASET" "$OUT"
python3 bonus/scripts/plot_eda.py "$OUT"
