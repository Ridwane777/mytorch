#!/bin/sh
# Starts, or resumes after a reboot or a crash, a training run in the background.
#   sh scripts/resume_training.sh [CONFIG [RUN_DIRECTORY]]
# Defaults: configs/cnn_network.conf and runs/cnn_full. BATCH (default 384) and the other overrides
# of train_final.sh (EPOCHS, PATIENCE, PROGRESS_EVERY) pass through. train_final.sh resumes from
# RUN/checkpoint.ckpt (ADR-0032), which needs the same settings as the first launch: only EPOCHS
# may change. Output is appended to RUN.out. It does nothing if that run is already going.
set -eu

CONFIG="${1:-configs/cnn_network.conf}"
RUN="${2:-runs/cnn_full}"
BATCH="${BATCH:-384}"
export BATCH
# One spelling per run directory (no trailing slash, relative to the root), so that the checks below
# and the name of the output file do not depend on how it was typed.
while [ "$RUN" != "${RUN%/}" ]; do RUN="${RUN%/}"; done
case "$RUN" in
"$(pwd)"/*) RUN="${RUN#"$(pwd)"/}" ;;
./*) RUN="${RUN#./}" ;;
esac

[ -f scripts/train_final.sh ] || { echo "resume_training.sh: run it from the repository root" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "resume_training.sh: $CONFIG not found" >&2; exit 1; }
# A run already going on this directory: the analyzer (its --save option names the directory) or
# train_final.sh, which also covers its build, data and scoring steps.
if pgrep -f "my_torch_analyzer --train --save $RUN/" > /dev/null ||
    pgrep -f "train_final\.sh [^ ]+ $RUN( |\$)" > /dev/null; then
    echo "resume_training.sh: $RUN is already training"
    exit 0
fi
if [ -f "$RUN/checkpoint.ckpt" ]; then
    echo "resuming $RUN after $(sed -n '3s/^epoch //p' "$RUN/checkpoint.ckpt") finished epochs"
else
    echo "starting $RUN"
fi
mkdir -p "$(dirname "$RUN")"
nohup setsid sh scripts/train_final.sh "$CONFIG" "$RUN" >> "$RUN.out" 2>&1 < /dev/null &
echo "started in the background (BATCH=$BATCH); output in $RUN.out, epochs in $RUN/epochs.log"
