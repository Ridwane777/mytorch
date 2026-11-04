#!/bin/sh
# my_torch against PyTorch, same architecture, same data, same machine (#42, ADR-0040,
# docs/performance/22-pytorch-reference.md). Needs comparison/.venv (see comparison/README.md):
# PyTorch lives only there, never in the library or in what the Makefile builds (ADR-0036).
#   gradient  one batch with identical weights: our gradients against autograd, at the initial
#             weights and after WARMUP Adam steps (comparison/gradient_check.py);
#   sweep     seconds per epoch against the number of threads, to give each implementation its best
#             thread count (the machine is shared with a desktop: 12 threads is not always best);
#   train     both configurations, FRACTION of the training part, EPOCHS epochs, scored on the whole
#             validation split (the test split is never read): my_torch, PyTorch from our initial
#             weights and batches (float64), PyTorch with its own initialisation and shuffling
#             (float64, then float32), several seeds for my_torch and PyTorch float64;
#   collect   tables and graphs (scripts/plot_pytorch_reference.py).
# Usage: sh scripts/bench_pytorch_reference.sh
# Environment: DATASET (default dataset), SEED (split seed, 42), RUN (default runs/pytorch_reference,
# large intermediate files, git-ignored), OUT (default docs/performance/data/pytorch_reference),
# STEPS (default "gradient sweep train collect"), FRACTION (0.1), EPOCHS (10), SEEDS_DENSE (5),
# SEEDS_CNN (3), THREADS_LIST ("1 2 4 6 8 12"), SWEEP_ROWS (4000), SWEEP_EPOCHS (3), VALID_ROWS (0 = the whole validation
# split; a smaller number is for trying the script out), GRAD_ROWS (64),
# WARMUP (100), MODELS ("dense cnn").
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
DATASET="${DATASET:-dataset}"
SEED="${SEED:-42}"
RUN="${RUN:-runs/pytorch_reference}"
OUT="${OUT:-docs/performance/data/pytorch_reference}"
STEPS="${STEPS:-gradient sweep train collect}"
FRACTION="${FRACTION:-0.1}"
EPOCHS="${EPOCHS:-10}"
SEEDS_DENSE="${SEEDS_DENSE:-5}"
SEEDS_CNN="${SEEDS_CNN:-3}"
THREADS_LIST="${THREADS_LIST:-1 2 4 6 8 12}"
SWEEP_ROWS="${SWEEP_ROWS:-4000}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-3}"
VALID_ROWS="${VALID_ROWS:-0}"
GRAD_ROWS="${GRAD_ROWS:-64}"
WARMUP="${WARMUP:-100}"
MODELS="${MODELS:-dense cnn}"
PY=comparison/.venv/bin/python
HELPER="$RUN/bin/pytorch_reference"

fail() {
    echo "bench_pytorch_reference.sh: $1" >&2
    exit 1
}
wants() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
config() { case "$1" in dense) echo configs/final_network.conf ;; cnn) echo configs/cnn_network.conf ;; esac; }
seeds_of() { case "$1" in dense) echo "$SEEDS_DENSE" ;; cnn) echo "$SEEDS_CNN" ;; esac; }
timed() { # file, command...: GNU time writes "seconds peak_kilobytes" to file
    f="$1"
    shift
    /usr/bin/time -f "%e %M" -o "$f" "$@"
}

mkdir -p "$RUN" "$OUT" "$RUN/data"
[ -x "$PY" ] || fail "$PY not found: create the virtualenv first (comparison/README.md)"

# 1. Build the my_torch side.
make -s build/libmy_torch.a
mkdir -p "$RUN/bin"
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude -Isrc/analyzer \
    scripts/pytorch_reference.cpp $(ls src/analyzer/*.cpp | grep -v main.cpp) build/libmy_torch.a \
    -o "$HELPER"

# 2. Data: the split of ADR-0017 (once), the first FRACTION of the training part, the whole
# validation part. Both configurations use fen-attacks-901: one export serves both.
if [ ! -s "$RUN/split/train.txt" ]; then
    OUT="$RUN/split" SEED="$SEED" DATASET="$DATASET" sh scripts/dataset_split.sh \
        "$RUN/split_stats" > "$RUN/split.log"
fi
total=$(wc -l < "$RUN/split/train.txt")
keep=$(python3 -c "import math; print(max(1, math.floor($total * $FRACTION)))")
head -n "$keep" "$RUN/split/train.txt" > "$RUN/data/train.txt"
if [ "$VALID_ROWS" -gt 0 ]; then
    head -n "$VALID_ROWS" "$RUN/split/validation.txt" > "$RUN/data/valid.txt"
else
    cp "$RUN/split/validation.txt" "$RUN/data/valid.txt"
fi
head -n "$SWEEP_ROWS" "$RUN/data/train.txt" > "$RUN/data/sweep_train.txt"
head -n 1000 "$RUN/data/valid.txt" > "$RUN/data/sweep_valid.txt"
for part in train valid sweep_train sweep_valid; do
    "$HELPER" export configs/final_network.conf "$RUN/data/$part.txt" "$RUN/data/$part" > /dev/null
done

# 3. Gradients against autograd.
if wants gradient; then
    : > "$RUN/gradient_check.log"
    rm -f "$OUT/gradient_check.csv"
    for model in $MODELS; do
        for state in init warm; do
            dir="$RUN/gradient/${model}_$state"
            mkdir -p "$dir"
            if [ "$state" = warm ]; then steps="$WARMUP"; else steps=0; fi
            "$HELPER" gradient "$(config "$model")" "$RUN/data/train.txt" "$GRAD_ROWS" "$dir" "$steps"
            "$PY" -W ignore comparison/gradient_check.py "$(config "$model")" "$dir" \
                "$dir/check.csv" "${model}_$state" | tee -a "$RUN/gradient_check.log"
        done
    done
    python3 scripts/plot_pytorch_reference.py gradient "$RUN" "$OUT" $MODELS
fi

# 4. Thread sweep: seconds per epoch on SWEEP_ROWS rows, SWEEP_EPOCHS epochs.
if wants sweep; then
    for model in $MODELS; do
        for t in $THREADS_LIST; do
            d="$RUN/sweep/$model/my_torch_$t"
            mkdir -p "$d"
            MY_TORCH_THREADS="$t" "$HELPER" train "$(config "$model")" "$RUN/data/sweep_train.txt" \
                "$RUN/data/sweep_valid.txt" "$SWEEP_EPOCHS" "$d" 2> /dev/null
            for dtype in float64 float32; do
                d="$RUN/sweep/$model/torch_${dtype}_$t"
                mkdir -p "$d"
                "$PY" comparison/train_reference.py --config "$(config "$model")" \
                    --train "$RUN/data/sweep_train" --valid "$RUN/data/sweep_valid" \
                    --epochs "$SWEEP_EPOCHS" --variant sweep --model "$model" --dtype "$dtype" \
                    --init native --order native --threads "$t" --out-dir "$d" 2> /dev/null
            done
            echo "sweep $model, $t threads done"
        done
    done
    python3 scripts/plot_pytorch_reference.py sweep "$RUN" "$OUT" $MODELS
fi

# 5. Training runs, each under GNU time (elapsed seconds and peak resident set size).
if wants train; then
    for model in $MODELS; do
        cfg=$(config "$model")
        t_mine=$(python3 scripts/plot_pytorch_reference.py best-threads "$OUT/thread_sweep.csv" "$model" my_torch)
        t_f64=$(python3 scripts/plot_pytorch_reference.py best-threads "$OUT/thread_sweep.csv" "$model" torch_float64)
        t_f32=$(python3 scripts/plot_pytorch_reference.py best-threads "$OUT/thread_sweep.csv" "$model" torch_float32)
        echo "$model threads: my_torch $t_mine, torch float64 $t_f64, torch float32 $t_f32"
        n=$(seeds_of "$model")
        k=0
        while [ "$k" -lt "$n" ]; do
            d="$RUN/train/$model/my_torch_s$k"
            mkdir -p "$d"
            cp /proc/loadavg "$d/load.txt"
            echo "threads $t_mine" > "$d/threads.txt"
            MY_TORCH_THREADS="$t_mine" timed "$d/time.txt" "$HELPER" train "$cfg" \
                "$RUN/data/train.txt" "$RUN/data/valid.txt" "$EPOCHS" "$d" "$k" 2> "$d/log.txt"
            echo "$model my_torch seed $k: $(tail -n 1 "$d/log.txt")"
            if [ "$k" -eq 0 ]; then
                # Same initial weights and batches as my_torch, float64.
                v="$RUN/train/$model/torch_shared_float64_s0"
                mkdir -p "$v"
                cp /proc/loadavg "$v/load.txt"
                echo "threads $t_f64" > "$v/threads.txt"
                timed "$v/time.txt" "$PY" comparison/train_reference.py --config "$cfg" \
                    --train "$RUN/data/train" --valid "$RUN/data/valid" --epochs "$EPOCHS" \
                    --variant torch_shared_float64 --model "$model" --dtype float64 --init shared \
                    --order shared --threads "$t_f64" --out-dir "$v" --reference "$d" 2> "$v/log.txt"
                echo "$model torch shared float64: $(tail -n 1 "$v/log.txt")"
                # PyTorch alone in float32, its own initialisation and shuffling.
                v="$RUN/train/$model/torch_native_float32_s0"
                mkdir -p "$v"
                cp /proc/loadavg "$v/load.txt"
                echo "threads $t_f32" > "$v/threads.txt"
                timed "$v/time.txt" "$PY" comparison/train_reference.py --config "$cfg" \
                    --train "$RUN/data/train" --valid "$RUN/data/valid" --epochs "$EPOCHS" \
                    --variant torch_native_float32 --model "$model" --dtype float32 --init native \
                    --order native --threads "$t_f32" --out-dir "$v" 2> "$v/log.txt"
                echo "$model torch native float32: $(tail -n 1 "$v/log.txt")"
            fi
            # PyTorch float64, its own initialisation and shuffling: same precision as ours.
            v="$RUN/train/$model/torch_native_float64_s$k"
            mkdir -p "$v"
            cp /proc/loadavg "$v/load.txt"
            echo "threads $t_f64" > "$v/threads.txt"
            timed "$v/time.txt" "$PY" comparison/train_reference.py --config "$cfg" \
                --train "$RUN/data/train" --valid "$RUN/data/valid" --epochs "$EPOCHS" \
                --variant torch_native_float64 --model "$model" --dtype float64 --init native \
                --order native --threads "$t_f64" --out-dir "$v" --seed-index "$k" 2> "$v/log.txt"
            echo "$model torch native float64 seed $k: $(tail -n 1 "$v/log.txt")"
            k=$((k + 1))
        done
    done
fi

# 6. Tables, graphs and the run card.
if wants collect; then
    {
        echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!QA.md' || echo ' (uncommitted changes)')"
        echo "machine: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), $(nproc) threads, $(free -m | awk '/Mem:/ {print $2}') MB RAM"
        echo "kernel: $(uname -sr)"
        echo "compiler: $(g++ --version | head -n 1)"
        echo "python: $($PY --version 2>&1), $($PY -c 'import torch, numpy; print("torch", torch.__version__ + ", numpy", numpy.__version__)')"
        echo "split seed: $SEED, training part: $total lines, kept: $keep ($FRACTION), validation: $(wc -l < "$RUN/data/valid.txt") lines, test split not read"
        echo "epochs: $EPOCHS, seeds: dense $SEEDS_DENSE, cnn $SEEDS_CNN, gradient batch: $GRAD_ROWS rows, warm-up: $WARMUP steps"
        echo "sweep: $SWEEP_ROWS rows, $SWEEP_EPOCHS epochs, threads $THREADS_LIST"
        echo "load average (1, 5, 15 min) when collecting: $(cut -d' ' -f1-3 /proc/loadavg)"
    } > "$OUT/run.txt"
    python3 scripts/plot_pytorch_reference.py collect "$RUN" "$OUT" $MODELS
fi
