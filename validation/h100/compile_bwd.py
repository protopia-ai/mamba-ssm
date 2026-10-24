"""Compile (lower and build, no launch) one Mamba-3 MIMO backward kernel variant and print a one-line result.

Compilation needs no input tensors, so this checks every configuration even where a launch would not fit the GPU.

Usage: python compile_bwd.py <kernel> N P R CHUNK BB_THREADS <variant>
  kernel:  bwd_fwd | bwd_bwd | varlen_bwd_fwd | varlen_bwd_bwd
  variant: reduceO | prereduce | fusednorm
"""

import sys
import traceback

import torch

from mamba_ssm.ops.tilelang.mamba3 import mamba3_mimo_bwd as bwd
from mamba_ssm.ops.tilelang.mamba3 import mamba3_mimo_bwd_varlen as bwd_varlen

B, S, H, G = 4, 2048, 16, 1  # FIXED_* values from tests/ops/tilelang/test_mamba3_mimo.py
ROTARY_DIM_DIVISOR = 4
DTYPE = "bfloat16"


def main() -> None:
    kernel, n, p, r, chunk, bb_threads, variant = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
    has_z, has_d = variant != "fusednorm", True
    reduce_o = variant != "prereduce"
    fuse = variant == "fusednorm"
    tag = f"{kernel} N{n}_P{p}_R{r}_C{chunk}_BB{bb_threads} {variant}"
    try:
        if kernel == "bwd_fwd":
            bwd.mamba_mimo_bwd_fwd(B, S, H, G, n, p, r, True, has_d, reduce_o, fuse, chunk, ROTARY_DIM_DIVISOR, DTYPE, 1e-5, 128, 0,
                                   states_dtype=torch.bfloat16)
        elif kernel == "bwd_bwd":
            bwd.mamba_mimo_bwd_bwd(B, S, H, G, n, p, r, has_z, has_d, reduce_o and not fuse, fuse, chunk, ROTARY_DIM_DIVISOR, DTYPE,
                                   bb_threads, 0, states_dtype=torch.bfloat16)
        elif kernel == "varlen_bwd_fwd":
            bwd_varlen.mamba_mimo_bwd_fwd(1, H, G, n, p, r, True, has_d, reduce_o, fuse, isVarlen=True, chunk_size=chunk,
                                          rotary_dim_divisor=ROTARY_DIM_DIVISOR, dtype=DTYPE, outproj_norm_eps=1e-5, threads=128,
                                          num_stages=0, states_dtype=torch.bfloat16)
        elif kernel == "varlen_bwd_bwd":
            bwd_varlen.mamba_mimo_bwd_bwd(1, H, G, n, p, r, has_z, has_d, reduce_o and not fuse, fuse, isVarlen=True, chunk_size=chunk,
                                          rotary_dim_divisor=ROTARY_DIM_DIVISOR, dtype=DTYPE, threads=bb_threads, num_stages=0,
                                          states_dtype=torch.bfloat16)
        else:
            raise ValueError(kernel)
        print(f"OK    {tag}")
    except Exception as e:  # noqa: BLE001 -- report any compile failure as a result line rather than crash the matrix
        lines = str(e).strip().splitlines()
        reason = next((line for line in lines if "Check failed" in line or "Error" in line), lines[-1] if lines else repr(e))
        print(f"FAIL  {tag} :: {type(e).__name__}: {reason[:300]}")
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
