#!/bin/sh
# Defense demo (docs/defense/demo.md): every step of the project in a few minutes, without the
# the school dataset. Each step waits for Enter; DEMO_AUTO=1 runs them all without pausing.
# Usage: sh scripts/demo.sh
set -eu
export LC_ALL=C

cd "$(dirname "$0")/.."
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
POSITIONS=docs/defense/demo_positions.txt

step() {
    printf '\n\033[1m== %s\033[0m\n' "$1"
    [ "${DEMO_AUTO:-0}" = 1 ] || { printf '(Enter)'; read -r _; }
}
run() { printf '$ %s\n' "$*"; "$@"; }

step "1. Build: the executables and the bonus tools (oracle, position generator)"
run make -s
run make -s -C bonus

step "2. The pushed network on ten famous positions (expected label at the end of each line)"
run cat "$POSITIONS"
run ./my_torch_analyzer --predict my_torch_network_final.nn "$POSITIONS"

step "3. The rule-based oracle (bonus) checks the expected labels"
run bonus/my_torch_oracle "$POSITIONS"

step "4. Generate a blank network from a configuration"
run sed -n '/^input/,$p' configs/basic_network.conf
(cd "$WORK" && run "$OLDPWD/my_torch_generator" "$OLDPWD/configs/basic_network.conf" 1)

step "5. Train it on 3,000 positions from our own generator (no stalemate), then predict"
bonus/my_torch_positions 3000 1 0 > "$WORK/train.txt" 2> /dev/null
run head -3 "$WORK/train.txt"
run ./my_torch_analyzer --train --save "$WORK/trained.nn" "$WORK/basic_network_1.nn" "$WORK/train.txt"
run ./my_torch_analyzer --predict "$WORK/trained.nn" "$POSITIONS"

step "6. Errors: message on standard error, exit code 84"
./my_torch_analyzer --predict missing.nn "$POSITIONS" || echo "exit code: $?"
./my_torch_analyzer --predict my_torch_network_final.nn missing.txt || echo "exit code: $?"
echo "not a fen" > "$WORK/bad.txt"
./my_torch_analyzer --predict my_torch_network_final.nn "$WORK/bad.txt" || echo "exit code: $?"

step "7. The library alone: tests, including the finite-difference gradient checks"
run make -s tests_run 2>&1 | tail -1
printf '\nEnd of the demo.\n'
