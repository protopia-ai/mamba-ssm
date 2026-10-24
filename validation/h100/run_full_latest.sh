#!/usr/bin/env bash
# Run the rest of the test suite (beyond the Mamba-3 suites) in the latest env, one group per GPU.
# Usage: run_full_latest.sh <group> <gpu>   groups: ssd | triton | toplevel | scan
# Writes results/full-latest-<group>.log and prints a one-line status.
set -uo pipefail
source "$(dirname "$0")/common.sh"
group=$1 gpu=$2
case "$group" in
    ssd) tests=(tests/ops/triton/test_ssd.py) ;;
    triton) tests=(tests/ops/triton/test_torch_compile.py tests/ops/triton/test_layer_norm.py tests/ops/triton/test_layernorm_gated.py
                   tests/ops/triton/test_k_activations.py tests/ops/triton/test_selective_state_update.py) ;;
    toplevel) tests=(tests/test_determinism.py tests/test_generation.py) ;;
    scan) tests=(tests/ops/test_selective_scan.py tests/modules) ;;
    *) echo "unknown group: $group" >&2; exit 2 ;;
esac
cd "$(src_for latest)"
env CUDA_VISIBLE_DEVICES="$gpu" $(env_vars_for latest) "$(py_for latest)" -m pytest -p no:cacheprovider -q -rfE "${tests[@]}" \
    > "$RESULTS/full-latest-$group.log" 2>&1
echo "$group exit=$? :: $(tail -1 "$RESULTS/full-latest-$group.log")"
