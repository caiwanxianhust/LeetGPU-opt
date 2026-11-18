#!/usr/bin/env python3
"""Per-kernel benchmark harness: one source of truth for inputs + measurement.

Usage:
    python3 bench.py <kernel_name>
    python3 bench.py <kernel_name> --ncu        # warmup + exactly 1 timed launch (for ncu -s/-c)
    python3 bench.py <kernel_name> --markdown   # emit a paste-ready BENCH.md row
    python3 bench.py <kernel_name> --ref torch  # also time the PyTorch reference
    python3 bench.py --list
    python3 bench.py <kernel_name> --print-test-fn

stdout is exactly ONE machine-readable CSV line:
    name,shape,dtype,median_ms,min_ms
Everything human-readable (spread, ref timing, warnings) goes to stderr, so
`python3 bench.py <k> > out.csv` stays clean.

Protocol: fixed seed -> WARMUP launches -> ITERS timed launches -> median/min.
Timing uses CUDA events (device time), same call path as test.py (including the
kernel's own internal output allocation). Requires CUDA, i.e. Full access
(see AGENTS.md rule 4).
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import torch

try:
    import leetgpu
except ImportError as exc:  # pragma: no cover
    sys.exit(f"cannot import leetgpu: {exc}\n"
             f"build first: python3 setup.py build_ext --inplace")

SEED = 0
WARMUP = 5
ITERS = 20
HERE = Path(__file__).resolve().parent


def _t(*shape, dtype=torch.float32):
    return torch.randn(*shape, device="cuda", dtype=dtype)


# --------------------------------------------------------------------------
# Kernel registry.
# One entry per kernel: pybind name -> test.py function name, input builder,
# dtype. The builder returns (run, ref_or_None, shape_str); `ref` is the
# PyTorch reference used by --ref, mirroring test.py's reference exactly.
# shape/dtype MUST match the test.py twin — that is the whole point of the
# "measurement path must not change" rule.
# --------------------------------------------------------------------------
def _fused_ln_linear_gelu():
    M = K = N = 4096
    x = _t(M, K)
    gamma = _t(N)
    beta = _t(N)
    W = _t(K, N)
    bias = _t(N)
    eps = 1e-5

    def run():
        return leetgpu.fused_ln_linear_gelu(x, gamma, beta, W, bias, eps)

    def ref():
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) * gamma / torch.sqrt(var + eps) + beta
        return torch.nn.functional.gelu(x_norm @ W + bias)

    return run, ref, f"{M}x{K}x{N}", "fp32"


KERNELS = {
    "fused_ln_linear_gelu": {
        "test_fn": "fused_layernorm_linear_gelu",  # test.py uses a longer name
        "build": _fused_ln_linear_gelu,
        "dtype": "fp32",
    },
}


def _resolve(name):
    """Accept either the pybind name or the test.py name."""
    if name in KERNELS:
        return name, KERNELS[name]
    for canonical, spec in KERNELS.items():
        if spec.get("test_fn") == name:
            return canonical, spec
    return None, None


def _time_once(run):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = run()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    del out
    return elapsed_ms


def _measure(run, iters):
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()
    samples = sorted(_time_once(run) for _ in range(iters))
    return samples


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _markdown_row(name, shape, dtype, median_ms, min_ms, vs_pytorch):
    return (f"| {date.today().isoformat()} | {_git_commit()} | {name} | {shape} "
            f"| {dtype} | {median_ms:.3f} | {min_ms:.3f} | — | {vs_pytorch} | — |")


def main(argv):
    ap = argparse.ArgumentParser(add_help=True, description=__doc__.splitlines()[0])
    ap.add_argument("kernel", nargs="?")
    ap.add_argument("--list", action="store_true", help="list supported kernels")
    ap.add_argument("--ncu", action="store_true",
                    help="warmup then a single launch (profile with: ncu -s %d -c 1)" % WARMUP)
    ap.add_argument("--markdown", action="store_true", help="emit a BENCH.md row")
    ap.add_argument("--ref", choices=["torch"], help="also time the PyTorch reference")
    ap.add_argument("--print-test-fn", action="store_true",
                    help="print the test.py function name and exit")
    args = ap.parse_args(argv[1:])

    if args.list:
        for name, spec in KERNELS.items():
            alias = spec.get("test_fn")
            sys.stderr.write(f"{name}\t(dtype={spec['dtype']}, test_fn={alias})\n")
        return 0

    if not args.kernel:
        ap.print_usage(sys.stderr)
        return 2

    canonical, spec = _resolve(args.kernel)
    if canonical is None:
        sys.stderr.write(f"unknown kernel: {args.kernel}\n"
                         f"supported: {', '.join(sorted(KERNELS))}\n")
        return 2

    if args.print_test_fn:
        fn = spec.get("test_fn") or ""
        if not fn:
            sys.stderr.write(f"{canonical} has no test_fn registered\n")
            return 2
        print(fn)
        return 0

    if not torch.cuda.is_available():
        sys.stderr.write(
            "CUDA unavailable. Under the workspace-write sandbox this shows up as "
            "`CUDA error 304: OS call failed` while nvidia-smi works; rerun with "
            "Full access (AGENTS.md rule 4).\n")
        return 3

    torch.cuda.synchronize()
    torch.manual_seed(SEED)
    run, ref, shape, dtype = spec["build"]()

    if args.ncu:
        for _ in range(WARMUP):
            run()
        torch.cuda.synchronize()
        elapsed_ms = _time_once(run)
        median_ms = min_ms = elapsed_ms
    else:
        samples = _measure(run, ITERS)
        n = len(samples)
        median_ms = (samples[n // 2] if n % 2
                     else (samples[n // 2 - 1] + samples[n // 2]) / 2)
        min_ms = samples[0]
        max_ms = samples[-1]
        mean = sum(samples) / n
        std = (sum((s - mean) ** 2 for s in samples) / n) ** 0.5
        sys.stderr.write(
            f"# seed={SEED} warmup={WARMUP} iters={ITERS} "
            f"median={median_ms:.3f}ms min={min_ms:.3f}ms max={max_ms:.3f}ms "
            f"spread={max_ms - min_ms:.3f}ms stddev={std:.3f}ms\n")

    vs_pytorch = "待测"
    if args.ref == "torch":
        if ref is None:
            sys.stderr.write(f"# no torch reference registered for {canonical}\n")
        else:
            ref_samples = _measure(ref, ITERS)
            ref_median = ref_samples[len(ref_samples) // 2]
            ratio = ref_median / median_ms if median_ms else float("inf")
            vs_pytorch = f"{ratio:.1f}x"
            sys.stderr.write(
                f"# ref(torch): median={ref_median:.3f}ms min={ref_samples[0]:.3f}ms "
                f"speedup={ratio:.2f}x\n")

    if args.markdown:
        sys.stderr.write("# markdown row (append to BENCH.md, fill vs-上一版/改动)\n")
        print(_markdown_row(canonical, shape, dtype, median_ms, min_ms, vs_pytorch))
    else:
        print(f"{canonical},{shape},{dtype},{median_ms:.3f},{min_ms:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
