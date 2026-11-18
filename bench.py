#!/usr/bin/env python3
"""Per-kernel micro-benchmark harness (fixed seed, warmup + median).

Usage:
    python3 bench.py <kernel_name>
    python3 bench.py --list

stdout is exactly ONE machine-readable CSV line:
    name,shape,median_ms,min_ms

Human-readable extras (max/stddev, sanity warnings) go to stderr, so
`python3 bench.py <k> > out.csv` stays clean.

Protocol: fixed seed -> warmup WARMUP times -> ITERS timed runs -> median/min.
Per-run timing uses CUDA events (not wall clock), so it measures device time of
the launch plus the kernel's own internal allocation, same call path as test.py.

Requires CUDA, i.e. must run under Full access (see AGENTS.md rule 4).
"""

import statistics
import sys

import torch

try:
    import leetgpu
except ImportError as exc:  # pragma: no cover
    sys.exit(f"cannot import leetgpu: {exc}\nrun from the repo root after "
             f"`python3 setup.py build_ext --inplace`")

SEED = 0
WARMUP = 5
ITERS = 20

F32 = torch.float32


def _t(*shape):
    return torch.randn(*shape, device="cuda", dtype=F32)


def fused_ln_linear_gelu():
    """Mirrors test.py::fused_layernorm_linear_gelu (M=K=N=4096, eps=1e-5)."""
    M = K = N = 4096
    x = _t(M, K)
    gamma = _t(N)
    beta = _t(N)
    W = _t(K, N)
    bias = _t(N)
    eps = 1e-5

    def run():
        return leetgpu.fused_ln_linear_gelu(x, gamma, beta, W, bias, eps)

    return run, f"{M}x{K}x{N}"


# name -> zero-arg builder returning (callable, shape_str).
# Aliases are allowed; keep every entry's shape identical to its test.py twin.
KERNELS = {
    "fused_ln_linear_gelu": fused_ln_linear_gelu,       # bindings.cu name
    "fused_layernorm_linear_gelu": fused_ln_linear_gelu,  # test.py name
}


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


def bench(name):
    if name not in KERNELS:
        sys.stderr.write(
            f"unknown kernel: {name}\nsupported: {', '.join(sorted(KERNELS))}\n")
        return 2

    if not torch.cuda.is_available():
        sys.stderr.write(
            "CUDA unavailable. Under the workspace-write sandbox this manifests as "
            "`CUDA error 304: OS call failed` while nvidia-smi works; rerun with "
            "Full access (AGENTS.md rule 4).\n")
        return 3

    torch.cuda.synchronize()
    torch.manual_seed(SEED)
    run, shape = KERNELS[name]()

    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    samples = [_time_once(run) for _ in range(ITERS)]
    samples.sort()
    median_ms = statistics.median(samples)
    min_ms = samples[0]
    max_ms = samples[-1]
    std_ms = statistics.pstdev(samples)

    print(f"{name},{shape},{median_ms:.3f},{min_ms:.3f}")
    sys.stderr.write(
        f"# seed={SEED} warmup={WARMUP} iters={ITERS} "
        f"median={median_ms:.3f}ms min={min_ms:.3f}ms max={max_ms:.3f}ms "
        f"spread={max_ms - min_ms:.3f}ms stddev={std_ms:.3f}ms\n")
    return 0


def main(argv):
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        sys.stderr.write(f"usage: python3 bench.py <kernel_name>\n"
                         f"       python3 bench.py --list\n")
        return 2
    if argv[1] == "--list":
        for k in sorted(KERNELS):
            sys.stderr.write(f"{k}\n")
        return 0
    return bench(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
