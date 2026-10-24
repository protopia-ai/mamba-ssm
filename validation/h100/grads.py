"""Save and compare Mamba-3 MIMO backward gradients and timings across environments.

`save` runs mamba_mimo_bwd_combined (or its varlen variant) from whichever mamba_ssm the current interpreter
imports, on the exact inputs the test suite builds, and writes gradients, the fp32 reference, and timings.
`compare` reports, per gradient, each run's error against the reference and the two runs against each other.

Usage:
  python grads.py save    <bwd|bwd_varlen> N P R CHUNK BB_THREADS OUT.pt [--iters 30]
  python grads.py compare A.pt B.pt
"""

import argparse
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

GRADS = ["dq", "dk", "dv", "dA", "ddt", "dtrap", "dq_bias", "dk_bias", "dmimo_v", "dmimo_z", "dmimo_o", "dangles", "dD", "dz"]


def _tests_module(repo: Path):
    sys.path.insert(0, str(repo / "tests" / "ops" / "tilelang"))
    import test_mamba3_mimo  # noqa: PLC0415

    return test_mamba3_mimo


def save(args: argparse.Namespace) -> None:
    import mamba_ssm  # noqa: PLC0415

    repo = Path(mamba_ssm.__file__).resolve().parents[1]
    t = _tests_module(repo)
    n, p, r, chunk, bb = args.n, args.p, args.r, args.chunk, args.bb_threads
    if args.kernel == "bwd":
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo as top  # noqa: PLC0415
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd as bwd  # noqa: PLC0415
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd as fwd  # noqa: PLC0415
        import mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils as utils  # noqa: PLC0415

        mods = SimpleNamespace(top=top, bwd=bwd, fwd=fwd, utils=utils)
        inputs = t.build_inputs(mods=mods, n=n, p=p, r=r, chunk_size=chunk, seed=5678 + n + p + r + chunk)
        ref = t.run_ref_backward_fp32(mods, inputs)
        fn, extra = bwd.mamba_mimo_bwd_combined, {}
    else:
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_bwd_varlen as bwd_varlen  # noqa: PLC0415
        import mamba_ssm.ops.tilelang.mamba3.mamba3_mimo_fwd_varlen as fwd_varlen  # noqa: PLC0415
        import mamba_ssm.ops.triton.mamba3.mamba3_mimo_utils as utils  # noqa: PLC0415

        mods = SimpleNamespace(fwd_varlen=fwd_varlen, bwd_varlen=bwd_varlen, utils_varlen=utils)
        inputs = t.build_inputs_varlen(
            mods_varlen=mods, n=n, p=p, r=r, chunk_size=chunk, seqlens=t._varlen_seqlens(chunk), seed=9002 + n + p + r + chunk
        )
        ref = t.run_ref_backward_fp32_varlen(inputs)
        fn, extra = bwd_varlen.mamba_mimo_bwd_combined_varlen, {"cu_seqlens": inputs["cu_seqlens"]}

    def call():
        return fn(
            inputs["dout"], inputs["q"], inputs["k"], inputs["v"], inputs["q_bias"], inputs["k_bias"], inputs["mimo_v"],
            inputs["mimo_o"], inputs["z"], inputs["mimo_z"], inputs["angles"], inputs["dA_cs"], inputs["dA_cs_rev"], inputs["dt"],
            inputs["trap"], inputs["D"], inputs["segsum"], chunk, inputs["rotary_dim_divisor"], t.FIXED_DTYPE, bb_threads=bb, **extra,
        )

    grads = dict(zip(GRADS, call()))
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(args.iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    out = {
        "config": f"{args.kernel} N{n}_P{p}_R{r}_C{chunk}_BB{bb}",
        "grads": {g: v.detach().cpu() for g, v in grads.items() if v is not None},
        "ref": {g: ref[g].detach().cpu() for g in GRADS if ref.get(g) is not None},
        "times_ms": times,
    }
    torch.save(out, args.out)
    print(f"saved {out['config']} median {statistics.median(times):.3f} ms -> {args.out}")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-5)).item()


def compare(args: argparse.Namespace) -> None:
    a, b = torch.load(args.a), torch.load(args.b)
    assert a["config"] == b["config"], (a["config"], b["config"])
    print(f"## {a['config']}")
    print(f"{'grad':8s} {'A_vs_ref':>10s} {'B_vs_ref':>10s} {'B_vs_A':>10s} bitwise")
    worst = 0.0
    for g in GRADS:
        if g not in a["grads"]:
            continue
        ga, gb, rf = a["grads"][g], b["grads"][g], a["ref"].get(g)
        ra = _rel(ga, rf) if rf is not None else float("nan")
        rb = _rel(gb, rf) if rf is not None else float("nan")
        rab = _rel(gb, ga)
        worst = max(worst, rab)
        print(f"{g:8s} {ra:10.3e} {rb:10.3e} {rab:10.3e} {torch.equal(ga, gb)}")
    ma, mb = statistics.median(a["times_ms"]), statistics.median(b["times_ms"])
    print(f"median ms: A {ma:.3f}  B {mb:.3f}  B/A {mb / ma:.3f}")
    print(f"max B_vs_A rel: {worst:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("save")
    s.add_argument("kernel", choices=["bwd", "bwd_varlen"])
    for name in ("n", "p", "r", "chunk", "bb_threads"):
        s.add_argument(name, type=int)
    s.add_argument("out")
    s.add_argument("--iters", type=int, default=30)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args()
    save(args) if args.cmd == "save" else compare(args)


if __name__ == "__main__":
    main()
