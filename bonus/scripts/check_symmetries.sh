#!/bin/sh
# Checks the board symmetries of src/analyzer/Symmetry.cpp (#38, ADR-0019) with the rule-based
# oracle: the augmented labels must be what the oracle computes on the transformed positions.
#   1. labelled positions: a FILE ("FEN label" per line, e.g. a dataset file), or N positions
#      generated and labelled by the oracle (my_torch_positions, 30 % stalemates);
#   2. scripts/augment.cpp writes the original, the mirror, the flip and the mirror+flip of each;
#   3. my_torch_oracle labels the 4 x N positions from scratch and compares them with the labels the
#      symmetries computed. Exit 0 only if "disagree: 0".
# Usage: sh bonus/scripts/check_symmetries.sh [N | FILE]   (default N = 2000)
set -eu

cd "$(dirname "$0")/../.."
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

make -s -C bonus
g++ -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude -Isrc/analyzer scripts/augment.cpp \
    src/analyzer/Fen.cpp src/analyzer/Symmetry.cpp -o "$WORK/augment"

SOURCE="${1:-2000}"
if [ -f "$SOURCE" ]; then
    cp "$SOURCE" "$WORK/positions.txt"
else
    bonus/my_torch_positions "$SOURCE" 1 0.3 > "$WORK/positions.txt" 2> "$WORK/generation.log"
fi
"$WORK/augment" --all "$WORK/positions.txt" "$WORK/augmented.txt"
bonus/my_torch_oracle "$WORK/augmented.txt" | tee "$WORK/report.txt"

[ "$(sed -n 's/^disagree: //p' "$WORK/report.txt")" = "0" ] || {
    echo "check_symmetries.sh: the oracle disagrees with the augmented labels" >&2
    exit 1
}
