#!/bin/sh
# Trial runs of #17 (step 3, docs/performance/09-trials.md): trains every configuration of
# configs/trials/ on a FRACTION of the dataset with scripts/train_final.sh, PARALLEL at a time,
# scores each network on the validation split (EVAL=validation: the test split is never used to
# choose), then gathers the results in trials.csv and trials.png.
# Usage: sh scripts/trial_runs.sh [output directory (default docs/performance/data)]
# Environment: DATASET (default dataset), FRACTION (default 0.1), PARALLEL (default: number of
# cores), RUNS (default runs/trials), TRIALS (default configs/trials), MY_TORCH_THREADS (default 1:
# the runs already share the cores, ADR-0024).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
OUT="${1:-docs/performance/data}"
FRACTION="${FRACTION:-0.1}"
PARALLEL="${PARALLEL:-$(nproc)}"
RUNS="${RUNS:-runs/trials}"
TRIALS="${TRIALS:-configs/trials}"
export DATASET="${DATASET:-dataset}"
export MY_TORCH_THREADS="${MY_TORCH_THREADS:-1}"

make -s my_torch_analyzer my_torch_generator
mkdir -p "$RUNS" "$OUT"
start=$(date +%s)
# One run per configuration; the executables are built once above (BUILD=0).
ls "$TRIALS"/*.conf | xargs -P "$PARALLEL" -I CONF sh -c '
    name=$(basename CONF .conf)
    FRACTION='"$FRACTION"' EVAL=validation BUILD=0 \
        sh scripts/train_final.sh CONF "'"$RUNS"'/$name" > "'"$RUNS"'/$name.out" 2>&1 \
        && echo "done: $name" || echo "FAILED: $name (see '"$RUNS"'/$name.out)"'
echo "trials: $(ls "$TRIALS"/*.conf | wc -l) runs, $PARALLEL at a time, $(( $(date +%s) - start )) s" \
    > "$OUT/trials_run.txt"
python3 scripts/summarize_trials.py "$RUNS" "$OUT"
cat "$OUT/trials_run.txt"
