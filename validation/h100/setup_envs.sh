#!/usr/bin/env bash
# Build the two environments this validation compares.
#   base: upstream main before this change, on the last stack known to work (tilelang 0.1.8, tvm-ffi 0.1.9, quack 0.5.0, cutlass 4.5.2)
#   new:  this branch, installed from its own pyproject.toml with no overrides
#
# Requires: x86_64 Linux, uv, git, Python 3.13 (uv can download it), and an NVIDIA driver that supports CUDA 13
# (the PyPI torch 2.13.0 wheel is built for cu130). The causal-conv1d wheel below is cp313/x86_64 only.
set -euo pipefail
source "$(dirname "$0")/common.sh"

CAUSAL_CONV1D='causal-conv1d @ https://github.com/protopia-ai/causal-conv1d/releases/download/v1.6.2.post1.nemotron_ultra_training_long_context_par.1/causal_conv1d-1.6.2.post1+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl'
COMMON=(torch==2.13.0 triton==3.7.1 "$CAUSAL_CONV1D" pytest einops transformers==5.18.0)

[ "$(uname -m)" = x86_64 ] || { echo "setup_envs.sh: the causal-conv1d wheel is x86_64 only; got $(uname -m)" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "setup_envs.sh: nvidia-smi not found" >&2; exit 1; }

if [ ! -d "$WORK/base-src" ]; then
    git -C "$REPO" fetch -q https://github.com/state-spaces/mamba.git "$BASE_REF" 2>/dev/null || true
    git -C "$REPO" worktree add --detach "$WORK/base-src" "$BASE_REF"
fi

printf '%s\n' "tilelang==0.1.8" "apache-tvm-ffi==0.1.9" "quack-kernels==0.5.0" \
    "nvidia-cutlass-dsl[cu13]==4.5.2" "nvidia-cutlass-dsl-libs-base==4.5.2" > "$WORK/base-overrides.txt"

uv venv -q --allow-existing -p 3.13 "$WORK/venv-base"
uv pip install -q -p "$(py_for base)" "${COMMON[@]}" -e "$WORK/base-src" --override "$WORK/base-overrides.txt"

uv venv -q --allow-existing -p 3.13 "$WORK/venv-new"
uv pip install -q -p "$(py_for new)" "${COMMON[@]}" -e "$REPO"

for env in base new; do
    "$(py_for "$env")" -I - "$env" <<'EOF' | tee "$RESULTS/versions-$env.txt"
import importlib.metadata as md, sys, torch
pkgs = ["mamba-ssm", "tilelang", "apache-tvm-ffi", "quack-kernels", "nvidia-cutlass-dsl", "torch", "triton", "causal-conv1d"]
print(sys.argv[1], {p: md.version(p) for p in pkgs})
print(sys.argv[1], "torch.cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
import mamba_ssm; print(sys.argv[1], "mamba_ssm from", mamba_ssm.__file__)
EOF
done
nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv | tee "$RESULTS/gpus.txt"
