#!/usr/bin/env bash
# Shared paths and helpers for the H100 validation scripts. Sourced, not executed.
#
# WORK      where the base source worktree, venvs, caches and logs go (default: <repo>/../mamba-h100-validation)
# BASE_REF  upstream commit the "base" environment is built from (default: upstream main before this change)

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
WORK=${WORK:-$(cd "$REPO/.." && pwd)/mamba-h100-validation}
BASE_REF=${BASE_REF:-e9594ce1c732d97440f0332fdc43170a2294dbfa}
RESULTS=$HERE/results
mkdir -p "$WORK" "$RESULTS"

# Source tree each environment imports mamba_ssm from.
src_for() {
    case "$1" in
        base) echo "$WORK/base-src" ;;
        new) echo "$REPO" ;;
        *) echo "unknown env: $1" >&2; return 1 ;;
    esac
}

py_for() { echo "$WORK/venv-$1/bin/python"; }

# Per-environment kernel caches so the two stacks never share compiled artifacts.
env_vars_for() {
    local c=$WORK/cache-$1
    echo "TRITON_CACHE_DIR=$c/triton TILELANG_CACHE_DIR=$c/tilelang CUTE_DSL_CACHE_DIR=$c/cute XDG_CACHE_HOME=$c/xdg"
}
