#!/bin/bash
# Fails if a neural-network or machine-learning library is used outside bonus/comparaison/ (ADR-0036).
# The subject forbids such libraries in the project; they are tolerated only in bonus/comparaison/,
# a separate directory with its own virtualenv, never built or called by the Makefile.
#
# Checks the files tracked by git (git grep): an untracked file is not seen, `git add` it first.
# Looked at: *.py *.cpp *.hpp *.h *.sh Makefile requirements*.txt, outside bonus/comparaison/ and this
# script itself. Matched: Python `import X` / `from X import ...` for torch, tensorflow, keras,
# sklearn, xgboost, lightgbm, jax; C and C++ includes of torch and tensorflow headers.
# Usage: scripts/check_comparison_isolation.sh
# Exit code: 0 when clean, 1 with the offending lines on stderr.
set -eu

cd "$(dirname "$0")/.."

LIBS='torch|tensorflow|keras|sklearn|xgboost|lightgbm|jax'
PYTHON_PATTERN="^[[:space:]]*(import|from)[[:space:]]+(${LIBS})([[:space:]]|\\.|,|;|\$)"
INCLUDE_PATTERN='^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"](torch|tensorflow)[/.]'
REQUIREMENT_PATTERN="^[[:space:]]*(${LIBS}|scikit-learn)([[:space:]]|[=<>~!\\[;]|\$)"

SELF="scripts/check_comparison_isolation.sh"
EXCLUDE=(':(exclude)bonus/comparaison' ":(exclude)${SELF}")

found=0
report() {
    if [ -n "$1" ]; then
        printf '%s\n' "$1" >&2
        found=1
    fi
}

report "$(git grep -n -I -E -e "$PYTHON_PATTERN" -- '*.py' '*.sh' 'Makefile' "${EXCLUDE[@]}" || true)"
report "$(git grep -n -I -E -e "$INCLUDE_PATTERN" -- '*.cpp' '*.hpp' '*.h' "${EXCLUDE[@]}" || true)"
report "$(git grep -n -I -E -e "$REQUIREMENT_PATTERN" -- 'requirements*.txt' '**/requirements*.txt' "${EXCLUDE[@]}" || true)"

if [ "$found" -ne 0 ]; then
    echo "check_comparison_isolation.sh: ML library used outside bonus/comparaison/ (ADR-0036)" >&2
    exit 1
fi
echo "check_comparison_isolation.sh: OK, no ML library outside bonus/comparaison/"
