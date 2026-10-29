#!/bin/sh
# Architecture diagrams of the dense and the convolutional networks (docs/performance/20-cnn.md):
#   1. builds the library and scripts/architecture.cpp, which loads a configuration (or a network
#      file) and prints one TSV line per layer: type, settings, output shape, parameter count, all
#      computed by the library (Layer::outputWidth and Layer::parameters, as modelSummary);
#   2. runs it on configs/final_network.conf and configs/cnn_network.conf;
#   3. draws the boxes with scripts/plot_architecture.py (matplotlib).
# Usage: sh scripts/architecture.sh
# Environment: OUT (default docs/performance/data), DENSE and CNN (the two configurations or
# network files to draw, default configs/final_network.conf and configs/cnn_network.conf).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
DEST="${OUT:-docs/performance/data}"
DENSE="${DENSE:-configs/final_network.conf}"
CNN="${CNN:-configs/cnn_network.conf}"
WORK=runs/architecture
mkdir -p "$WORK" "$DEST"

make -s build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/architecture.cpp \
    build/libmy_torch.a -o "$WORK/architecture"

"$WORK/architecture" "$DENSE" > "$DEST/architecture_dense.tsv"
"$WORK/architecture" "$CNN" > "$DEST/architecture_cnn.tsv"
python3 scripts/plot_architecture.py "$DEST/architecture_dense.tsv" "$DEST/architecture_dense.png" \
    "Dense network"
python3 scripts/plot_architecture.py "$DEST/architecture_cnn.tsv" "$DEST/architecture_cnn.png" \
    "Convolutional network"
