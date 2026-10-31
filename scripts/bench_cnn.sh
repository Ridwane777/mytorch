#!/bin/sh
# CNN against the dense network (#29, docs/performance/20-cnn.md): collects the measurements of two
# runs of scripts/train_final.sh, one per network, into docs/performance/data/cnn_vs_dense.csv, and
# draws docs/performance/data/cnn_vs_dense.png with scripts/plot_cnn.py.
#   - test accuracy, macro F1 and per-class precision, recall and F1: scores.txt (scripts/score.cpp);
#   - training time, epochs run, peak memory, machine: run.txt;
#   - prediction time: wall time of `my_torch_analyzer --predict` over the whole test split,
#     median of 3 runs made here, one after the other, output discarded;
#   - parameter count: scripts/architecture.cpp, which counts Layer::parameters() as modelSummary;
#   - file size: the saved network, network.nn.
# Usage: sh scripts/bench_cnn.sh CNN_RUN_DIRECTORY DENSE_RUN_DIRECTORY
# Run both trainings first, on the same machine, one at a time:
#   sh scripts/train_final.sh configs/cnn_network.conf runs/cnn
#   sh scripts/train_final.sh configs/final_network.conf runs/final_bench
# Environment: OUT (default docs/performance/data), REPEATS (prediction timings, default 3).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
[ $# -eq 2 ] || { echo "usage: sh scripts/bench_cnn.sh CNN_RUN DENSE_RUN" >&2; exit 1; }
CNN="$1"
DENSE="$2"
DEST="${OUT:-docs/performance/data}"
REPEATS="${REPEATS:-3}"
fail() {
    echo "bench_cnn.sh: $1" >&2
    exit 1
}
for run in "$CNN" "$DENSE"; do
    for f in run.txt scores.txt network.nn test_boards.txt; do
        [ -s "$run/$f" ] || fail "$run/$f not found: run scripts/train_final.sh first (EVAL=test)"
    done
    grep -q 'scored on: test' "$run/run.txt" || fail "$run was not scored on the test split"
done
mkdir -p "$DEST" runs/architecture

make -s my_torch_analyzer build/libmy_torch.a
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/architecture.cpp \
    build/libmy_torch.a -o runs/architecture/architecture

# median of the wall times of REPEATS predictions, in seconds
predict_time() {
    run="$1"
    i=0
    : > "$run/predict_repeats.txt"
    while [ "$i" -lt "$REPEATS" ]; do
        /usr/bin/time -f "%e" -a -o "$run/predict_repeats.txt" \
            ./my_torch_analyzer --predict "$run/network.nn" "$run/test_boards.txt" > /dev/null
        i=$((i + 1))
    done
    sort -n "$run/predict_repeats.txt" | awk '{v[NR]=$1} END {print v[int((NR + 1) / 2)]}'
}

score() { # file, line prefix, field
    sed -n "s/^$2: .*$3 \([0-9.]*\).*/\1/p" "$1" | head -1
}
global() { # file, key
    sed -n "s/^$2: \(.*\)/\1/p" "$1" | head -1
}
runfield() { # run, regular expression giving the number in the capture group
    sed -n "s/$2/\1/p" "$1/run.txt" | head -1
}

CSV="$DEST/cnn_vs_dense.csv"
cnn_predict=$(predict_time "$CNN")
dense_predict=$(predict_time "$DENSE")
{
    echo "metric,cnn,dense"
    row() { echo "$1,$2,$3"; }
    row test_positions "$(global "$CNN/scores.txt" 'positions scored')" "$(global "$DENSE/scores.txt" 'positions scored')"
    row accuracy "$(global "$CNN/scores.txt" accuracy)" "$(global "$DENSE/scores.txt" accuracy)"
    row macro_f1 "$(global "$CNN/scores.txt" 'macro F1')" "$(global "$DENSE/scores.txt" 'macro F1')"
    for class in "Nothing" "Check White" "Check Black" "Checkmate White" "Checkmate Black"; do
        key=$(echo "$class" | tr 'A-Z ' 'a-z_')
        for field in precision recall F1; do
            lower=$(echo "$field" | tr 'A-Z' 'a-z')
            row "${lower}_$key" "$(score "$CNN/scores.txt" "$class" "$field")" \
                "$(score "$DENSE/scores.txt" "$class" "$field")"
        done
        row "support_$key" "$(score "$CNN/scores.txt" "$class" support)" \
            "$(score "$DENSE/scores.txt" "$class" support)"
    done
    row train_seconds "$(runfield "$CNN" '^training: \([0-9.]*\) s.*')" "$(runfield "$DENSE" '^training: \([0-9.]*\) s.*')"
    row epochs_run "$(runfield "$CNN" '^epochs run: \([0-9]*\)')" "$(runfield "$DENSE" '^epochs run: \([0-9]*\)')"
    row train_peak_mb "$(runfield "$CNN" '^training: .*peak \([0-9]*\) MB')" "$(runfield "$DENSE" '^training: .*peak \([0-9]*\) MB')"
    row predict_seconds_median "$cnn_predict" "$dense_predict"
    row parameters "$(runs/architecture/architecture "$CNN/network.nn" | awk -F'\t' '$1 == "total" {print $5}')" \
        "$(runs/architecture/architecture "$DENSE/network.nn" | awk -F'\t' '$1 == "total" {print $5}')"
    row file_bytes "$(wc -c < "$CNN/network.nn" | tr -d ' ')" "$(wc -c < "$DENSE/network.nn" | tr -d ' ')"
} > "$CSV"

# The two run cards and scores, next to the CSV, so that the document can point at them.
for pair in "cnn:$CNN" "dense:$DENSE"; do
    name="${pair%%:*}"
    run="${pair#*:}"
    mkdir -p "$DEST/cnn-vs-dense/$name"
    cp "$run/run.txt" "$run/scores.txt" "$run/predict_repeats.txt" "$DEST/cnn-vs-dense/$name/"
    for f in curve.csv confusion.csv; do
        [ ! -f "$run/$f" ] || cp "$run/$f" "$DEST/cnn-vs-dense/$name/"
    done
done
cat "$CSV"
python3 scripts/plot_cnn.py "$CSV" "$DEST/cnn_vs_dense.png"
