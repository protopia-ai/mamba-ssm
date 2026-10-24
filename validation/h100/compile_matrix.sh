#!/usr/bin/env bash
# Compile every MIMO backward kernel variant the tests use, in one environment, 6 at a time.
# Usage: compile_matrix.sh <base|new> <gpu-index>
# Writes results/compile-<env>.txt with one OK/FAIL line per variant (132 total).
set -euo pipefail
source "$(dirname "$0")/common.sh"
env=$1 gpu=$2
out=$RESULTS/compile-$env.txt
: > "$out"
jobs=$WORK/compile-jobs.txt
: > "$jobs"
# (N, P, R, chunk, bwd_bwd threads) from CASE_GRID in tests/ops/tilelang/test_mamba3_mimo.py
for c in "16 64 4 8 128" "32 64 4 16 256" "64 64 4 16 256" "128 64 4 16 256" "256 64 4 8 256" "64 128 4 16 256" \
         "128 32 4 16 256" "128 128 4 8 256" "128 64 8 8 256" "128 64 2 32 256" "128 64 1 64 256"; do
    for k in bwd_fwd bwd_bwd varlen_bwd_fwd varlen_bwd_bwd; do
        for v in reduceO prereduce fusednorm; do echo "$k $c $v" >> "$jobs"; done
    done
done
cd "$(src_for "$env")"
export CUDA_VISIBLE_DEVICES=$gpu PY=$(py_for "$env") HERE out
nl -nln "$jobs" | xargs -P 6 -L1 bash -c '
    i=$0
    line=$(TILELANG_CACHE_DIR='"$WORK"'/cache-'"$env"'/compile-$i timeout 1800 "$PY" "$HERE/compile_bwd.py" "$@" 2>/dev/null | grep -E "^(OK|FAIL)")
    echo "${line:-FAIL  $* :: crashed or timed out}" >> "$out"'
sort -k2 -o "$out" "$out"
echo "$env compile: $(grep -c "^OK" "$out" || true) OK, $(grep -c "^FAIL" "$out" || true) FAIL of $(wc -l < "$jobs")"
