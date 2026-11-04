#!/bin/sh
# Final comparison report (#44, docs/performance/26-comparison.md).
#   1. the rule-based oracle (bonus/oracle) on the test part of the split, under GNU time;
#   2. bonus/comparaison/report.py: the measurements of 22 (PyTorch), 23 (baselines) and the oracle
#      in one table, the current network file sizes, the lines of code, and the graph.
# Needs the split written by scripts/bench_baselines.sh (RUN) and a Python with matplotlib (the
# default is the virtualenv of bonus/python/, see bonus/python/README.md).
# Usage: sh bonus/comparaison/comparison_report.sh
# Environment: RUN (default runs/baselines), OUT (default docs/performance/data/comparison),
# PY (default bonus/python/.venv/bin/python).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/../.."
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/comparison}"
PY="${PY:-bonus/python/.venv/bin/python}"

[ -s "$RUN/split/test.txt" ] || {
    echo "comparison_report.sh: $RUN/split/test.txt missing: run scripts/bench_baselines.sh" >&2
    exit 1
}
mkdir -p "$OUT/oracle"

make -s -C bonus my_torch_oracle
/usr/bin/time -f "%e %M" -o "$OUT/oracle/time.txt" bonus/my_torch_oracle "$RUN/split/test.txt" \
    > "$OUT/oracle/oracle.txt"
"$PY" bonus/comparaison/report.py . "$OUT/oracle" "$OUT" | tee "$OUT/summary.txt"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "commit: $(git rev-parse --short HEAD)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "threads: $(nproc)"
    echo "compiler: $(g++ --version | head -1)"
    echo "python: $("$PY" --version)"
} > "$OUT/run.txt"
