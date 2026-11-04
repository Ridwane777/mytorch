#!/bin/sh
# Keeps a copy of the training state of every epoch of a run that is already going, without
# touching it: train_final.sh writes RUN/checkpoint.ckpt after every epoch (ADR-0032), replacing
# the file atomically, and this script copies each new version.
#   sh scripts/watch_epochs.sh RUN_DIRECTORY [SECONDS_BETWEEN_CHECKS (default 20) [once]]
# train_final.sh starts it by itself. With "once" it makes one check and exits.
# For each finished epoch N it writes
#   RUN/epochs/checkpoint.<YYYYMMDD>-<HHMMSS>.epochN.ckpt   weights, optimizer and history: a
#                                                           training can resume from it
# and appends the epoch's measurements to RUN/epochs.log. It stops when RUN/run.txt exists (the
# run is over) or when it is interrupted. A copy never replaces another, and a restart does not
# copy again the epochs already copied.
set -eu
export LC_ALL=C

[ $# -ge 1 ] || { echo "usage: sh scripts/watch_epochs.sh RUN_DIRECTORY [SECONDS]" >&2; exit 1; }
RUN="$1"
EVERY="${2:-20}"
case "$EVERY" in
'' | *[!0-9]* | 0) echo "watch_epochs.sh: SECONDS must be a positive integer" >&2; exit 1 ;;
esac
[ -d "$RUN" ] || { echo "watch_epochs.sh: $RUN not found" >&2; exit 1; }
mkdir -p "$RUN/epochs"
# The last epoch already copied (0 if none), so that a restart does not copy it again.
last="$(ls "$RUN/epochs" 2>/dev/null | sed -n 's/^.*\.epoch\([0-9][0-9]*\)\.ckpt$/\1/p' |
    sort -n | tail -n 1)"
last="${last:-0}"

# The finished epochs of the checkpoint, from its third line ("epoch <finished epochs>").
finished() {
    sed -n '3s/^epoch //p' "$1" 2>/dev/null | head -n 1
}

copy_if_new() {
    [ -f "$RUN/checkpoint.ckpt" ] || return 0
    tmp="$RUN/epochs/.copy.$$"
    cp "$RUN/checkpoint.ckpt" "$tmp"
    n="$(finished "$tmp")"
    case "$n" in '' | *[!0-9]*) rm -f "$tmp"; return 0 ;; esac
    if [ "$n" -gt "$last" ]; then
        stamp="$(date +%Y%m%d-%H%M%S)"
        mv "$tmp" "$RUN/epochs/checkpoint.$stamp.epoch$n.ckpt"
        # trainLoss validationLoss penalty trainAccuracy validationAccuracy of the last epoch.
        awk -v n="$n" -v stamp="$stamp" '
            $1 == "history" { count = $2; start = NR }
            start && NR == start + count && count > 0 {
                printf "%s epoch %d  loss %.4f  accuracy %.4f  validation loss %.4f  " \
                       "validation accuracy %.4f\n", stamp, n, $1, $4, $2, $5
            }' "$RUN/epochs/checkpoint.$stamp.epoch$n.ckpt" | tee -a "$RUN/epochs.log"
        last="$n"
    else
        rm -f "$tmp"
    fi
}

if [ "${3:-}" = once ]; then
    copy_if_new
    exit 0
fi
while true; do
    copy_if_new
    [ ! -f "$RUN/run.txt" ] || break
    sleep "$EVERY"
done
copy_if_new
