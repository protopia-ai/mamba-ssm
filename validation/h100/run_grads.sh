#!/usr/bin/env bash
# Save gradients and timings for every test config in both environments, then compare them.
# Usage: run_grads.sh <base-gpu> <new-gpu>
# Writes results/grads/*.pt and results/grads.txt (A = base, B = new).
set -uo pipefail
source "$(dirname "$0")/common.sh"
gpu_base=$1 gpu_new=$2
mkdir -p "$RESULTS/grads"
: > "$RESULTS/grads.txt"
save() {  # env gpu kernel N P R C BB
    local env=$1 gpu=$2; shift 2
    local tag="$1-N$2_P$3_R$4_C$5_BB$6"
    (cd "$(src_for "$env")" && env CUDA_VISIBLE_DEVICES="$gpu" $(env_vars_for "$env") \
        "$(py_for "$env")" "$HERE/grads.py" save "$@" "$RESULTS/grads/$env-$tag.pt") \
        > "$RESULTS/grads/$env-$tag.log" 2>&1 || echo "SAVE FAILED: $env $tag (see results/grads/$env-$tag.log)" | tee -a "$RESULTS/grads.txt"
}
# CASE_GRID (bwd) and VARLEN_CASE_GRID (bwd_varlen) from tests/ops/tilelang/test_mamba3_mimo.py
configs=()
for c in "16 64 4 8 128" "32 64 4 16 256" "64 64 4 16 256" "128 64 4 16 256" "256 64 4 8 256" "64 128 4 16 256" \
         "128 32 4 16 256" "128 128 4 8 256" "128 64 8 8 256" "128 64 2 32 256" "128 64 1 64 256"; do configs+=("bwd $c"); done
for c in "16 64 4 8 128" "32 64 4 16 256" "64 64 4 16 256" "128 64 4 16 256" "128 64 2 32 256" "128 64 1 64 256"; do
    configs+=("bwd_varlen $c")
done
for cfg in "${configs[@]}"; do
    set -- $cfg
    save base "$gpu_base" "$@" &
    save new "$gpu_new" "$@" &
    wait
    tag="$1-N$2_P$3_R$4_C$5_BB$6"
    if [ -f "$RESULTS/grads/base-$tag.pt" ] && [ -f "$RESULTS/grads/new-$tag.pt" ]; then
        "$(py_for new)" -I "$HERE/grads.py" compare "$RESULTS/grads/base-$tag.pt" "$RESULTS/grads/new-$tag.pt" >> "$RESULTS/grads.txt" 2>&1
        echo >> "$RESULTS/grads.txt"
    fi
    echo "done $tag"
done
