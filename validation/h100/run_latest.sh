#!/usr/bin/env bash
# Validate the PR branch on the newest dependency stack (venv-latest) against the tilelang 0.1.8 base results.
# Usage: run_latest.sh [base-gpu] [latest-gpu] [grads-latest-gpu]   (defaults 0 1 2; venv-latest must already exist)
set -uo pipefail
source "$(dirname "$0")/common.sh"
gpu_base=${1:-0} gpu_latest=${2:-1} gpu_grads=${3:-2}
"$HERE/compile_matrix.sh" latest "$gpu_latest" | tee "$RESULTS/compile-latest-summary.txt"
"$HERE/run_suite.sh" latest "$gpu_latest" > "$RESULTS/suite-latest.status" &
"$HERE/run_grads_latest.sh" "$gpu_base" "$gpu_grads" > "$RESULTS/run_grads_latest.log" 2>&1 &
wait
cat "$RESULTS/suite-latest.status"
"$(py_for latest)" -I "$HERE/compare_suites.py" "$RESULTS/suite-base.log" "$RESULTS/suite-latest.log" | tee "$RESULTS/suite-diff-latest.txt"
echo "Results in $RESULTS"
