#!/bin/sh
# Stalemate bonus experiment (#24): can a network learn the Stalemate class from generated
# positions? Reproduces docs/performance/14-stalemate.md.
#   1. 5,000 generated positions (2,500 stalemates, 2,500 other sparse positions, oracle labels);
#   2. 20,000 dataset positions drawn with a fixed seed;
#   3. shuffled together, 80 % to train, 20 % to test (never trained on);
#   4. bonus/configs/stalemate_network.conf generated and trained with my_torch_analyzer;
#   5. predictions on the test set scored by bonus/scripts/score_predictions.py, which calls
#      scripts/score.cpp (the library's metrics, ADR-0026).
# Usage: sh bonus/scripts/stalemate_experiment.sh [output directory]
set -eu

cd "$(dirname "$0")/../.."
OUT="${1:-docs/performance/data}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
seeded() { yes "$1" | head -c 1048576; }

make -s
make -s -C bonus
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/score.cpp build/libmy_torch.a \
    -o "$WORK/score"
./bonus/my_torch_positions 5000 1 0.5 > "$WORK/generated.txt" 2> "$WORK/generation.log"
sed 's/$/ generated/' "$WORK/generated.txt" > "$WORK/generated_tagged.txt"
# shuf reads its randomness from a file: a fixed file gives a fixed draw.
seeded 1 > "$WORK/seed_a"
seeded 2 > "$WORK/seed_b"
cat dataset/check/*.txt dataset/checkmate/*.txt dataset/nothing/*.txt |
    shuf -n 20000 --random-source="$WORK/seed_a" > "$WORK/dataset.txt"
sed 's/$/ dataset/' "$WORK/dataset.txt" > "$WORK/dataset_tagged.txt"
cat "$WORK/generated_tagged.txt" "$WORK/dataset_tagged.txt" |
    shuf --random-source="$WORK/seed_b" > "$WORK/all.txt"
head -n 20000 "$WORK/all.txt" | sed 's/ [a-z]*$//' > "$WORK/train.txt"
tail -n 5000 "$WORK/all.txt" > "$WORK/test_tagged.txt"
sed 's/ [a-z]*$//' "$WORK/test_tagged.txt" > "$WORK/test.txt"
cut -d' ' -f1-6 "$WORK/test.txt" > "$WORK/test_boards.txt"

(cd "$WORK" && "$OLDPWD/my_torch_generator" "$OLDPWD/bonus/configs/stalemate_network.conf" 1)
/usr/bin/time -f "%e s, %M KB" -o "$WORK/train_time.txt" \
    ./my_torch_analyzer --train "$WORK/stalemate_network_1.nn" "$WORK/train.txt" \
    > "$WORK/training.log"
./my_torch_analyzer --predict "$WORK/stalemate_network_1.nn" "$WORK/test_boards.txt" \
    > "$WORK/predictions.txt"

mkdir -p "$OUT"
cp "$WORK/training.log" "$OUT/stalemate_training.log"
{ cat "$WORK/generation.log"; echo "training: $(cat "$WORK/train_time.txt")"; } \
    > "$OUT/stalemate_run.txt"
python3 bonus/scripts/score_predictions.py "$WORK/score" "$WORK/stalemate_network_1.nn" \
    "$WORK/test_tagged.txt" "$WORK/predictions.txt" "$OUT"
