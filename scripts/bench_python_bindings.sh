#!/bin/sh
# Python bindings against the C++ library (ADR-0044, docs/performance/25-python-bindings.md).
# Needs bonus/python/.venv with pybind11 and NumPy (bonus/python/README.md).
#   1. the test part of the split (ADR-0017, seed 42) encoded by `pytorch_reference export`, and
#      my_torch_analyzer's predictions on it: reused from scripts/bench_baselines.sh (RUN);
#   2. Network::predict timed in C++ (scripts/bench_python_bindings.cpp), probabilities saved;
#   3. the same through the Python module (bonus/python/bench.py): time, bit-for-bit comparison of
#      the probabilities, labels against the analyzer's;
#   4. the graph (scripts/plot_python_bindings.py).
# Usage: sh scripts/bench_python_bindings.sh
# Environment: RUN (default runs/baselines, from bench_baselines.sh), OUT (default
# docs/performance/data/python_bindings), NETWORK (default my_torch_network_final.nn),
# REPEATS (default 5).
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
RUN="${RUN:-runs/baselines}"
OUT="${OUT:-docs/performance/data/python_bindings}"
NETWORK="${NETWORK:-my_torch_network_final.nn}"
REPEATS="${REPEATS:-5}"
PY=bonus/python/.venv/bin/python
WORK="$RUN/python_bindings"

fail() {
    echo "bench_python_bindings.sh: $1" >&2
    exit 1
}

[ -x "$PY" ] || fail "$PY not found: create the virtualenv first (bonus/python/README.md)"
[ -s "$RUN/data/test.x.u8" ] || fail "$RUN/data/test.x.u8 missing: run scripts/bench_baselines.sh"
PREDICTIONS="$RUN/ours/$(basename "$NETWORK" .nn)/predictions.txt"
[ -s "$PREDICTIONS" ] || fail "$PREDICTIONS missing: run scripts/bench_baselines.sh"
ROWS=$(cut -d' ' -f1 "$RUN/data/test.shape")
mkdir -p "$WORK" "$OUT"

make -s build/libmy_torch.a
make -s -C bonus python PYTHON=python/.venv/bin/python
g++ -std=c++20 -O2 -pthread -Wall -Wextra -Werror -Iinclude scripts/bench_python_bindings.cpp \
    build/libmy_torch.a -o "$WORK/bench_python_bindings"

"$WORK/bench_python_bindings" "$NETWORK" "$RUN/data/test.x.u8" "$ROWS" "$REPEATS" \
    "$WORK/cpp.f64" > "$WORK/cpp_times.txt"
PYTHONPATH=bonus/python "$PY" bonus/python/bench.py "$NETWORK" "$RUN/data/test.x.u8" "$ROWS" \
    "$REPEATS" "$WORK/cpp.f64" "$PREDICTIONS" "$WORK/cpp_times.txt" "$OUT/times.csv" \
    | tee "$OUT/summary.txt"
{
    echo "date: $(date -u +%Y-%m-%dT%H:%MZ)"
    echo "commit: $(git rev-parse --short HEAD)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
    echo "threads: $(nproc)"
    echo "compiler: $(g++ --version | head -1)"
    echo "python: $("$PY" --version)"
    echo "pybind11: $("$PY" -c 'import pybind11; print(pybind11.__version__)')"
    echo "numpy: $("$PY" -c 'import numpy; print(numpy.__version__)')"
    echo "network: $NETWORK"
} > "$OUT/run.txt"
"$PY" scripts/plot_python_bindings.py "$OUT"
