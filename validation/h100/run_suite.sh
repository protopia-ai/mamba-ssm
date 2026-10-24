#!/usr/bin/env bash
# Run the Mamba-3 test suites (everything that touches tilelang, cutlass or quack) in one environment.
# Usage: run_suite.sh <base|new> <gpu-index>
# Writes results/suite-<env>.log; compare the two with compare_suites.py.
set -uo pipefail
source "$(dirname "$0")/common.sh"
env=$1 gpu=$2
cd "$(src_for "$env")"
env CUDA_VISIBLE_DEVICES="$gpu" $(env_vars_for "$env") "$(py_for "$env")" -m pytest -p no:cacheprovider -q -rA \
    tests/ops/tilelang tests/ops/cute tests/modules tests/ops/triton/test_mamba3_siso.py \
    > "$RESULTS/suite-$env.log" 2>&1
status=$?
echo "$env suite exit=$status :: $(tail -1 "$RESULTS/suite-$env.log")"
