#!/usr/bin/env bash
# Run the whole validation: environments, compile matrix, both test suites, and the gradient/timing comparison.
# Usage: run_all.sh [base-gpu] [new-gpu]   (defaults 0 and 1; the two environments run side by side)
set -uo pipefail
source "$(dirname "$0")/common.sh"
gpu_base=${1:-0} gpu_new=${2:-1}

"$HERE/setup_envs.sh" || exit 1
"$HERE/compile_matrix.sh" new "$gpu_new" | tee "$RESULTS/compile-summary.txt"
"$HERE/run_suite.sh" base "$gpu_base" > "$RESULTS/suite-base.status" &
"$HERE/run_suite.sh" new "$gpu_new" > "$RESULTS/suite-new.status" &
wait
cat "$RESULTS"/suite-*.status
"$(py_for new)" -I "$HERE/compare_suites.py" "$RESULTS/suite-base.log" "$RESULTS/suite-new.log" | tee "$RESULTS/suite-diff.txt"
"$HERE/run_grads.sh" "$gpu_base" "$gpu_new"
echo "Results in $RESULTS"
