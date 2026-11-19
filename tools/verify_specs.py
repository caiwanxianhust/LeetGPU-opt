"""Verify every registered Spec against the built extension.

Three jobs, all machine-checkable:

1. coverage -- every name in `src/bindings.cu`'s PYBIND11_MODULE has a Spec,
   and the two known benchmark-only kernels are reported as such;
2. correctness -- each Spec runs through `harness check` in its own process
   (so one crash cannot take the sweep down) and the PASS/FAIL/SKIP verdict is
   collected;
3. spot check (`--spot N`) -- for N specs, compare the kernel against its
   reference twice: once unchanged (must pass) and once with the candidate
   perturbed by 1% (must fail).  A Spec whose comparison accepts perturbed
   output is vacuous, and this is the only automated defence against that.

Usage:
    python3 tools/verify_specs.py                # coverage + correctness
    python3 tools/verify_specs.py --spot 12      # add the perturbation check
    python3 tools/verify_specs.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import checker, registry  # noqa: E402

# kernels that exist but have no legacy test (documented in README)
BENCHMARK_ONLY = {"adder_transformer_inference", "ssm_selective_scan"}

# Specs that reproduce a *pre-existing* kernel defect rather than a legacy
# pass: the legacy compare_tensors call reported the same mismatch and only
# exited 0 because its return value was dropped.  Listed here so the defect is
# visible instead of being papered over by a wider tolerance.
# Every entry was confirmed twice: (1) the legacy test itself prints a
# "Maximum difference" / "mismatched elements" line, and (2) substituting the
# Spec's reference for the kernel makes the legacy test pass, which proves the
# reference is faithful and the kernel is what disagrees with torch.
KNOWN_FAIL = {
    "W4A16_matmul":
        "fp16 accumulation order; max|diff|=1.0 against |ref|max~1900 "
        "(legacy reported the same mismatch)",
    "gpt2_transformer_block":
        "max|diff|~158-258 against tol=1.0; legacy also printed 98/98304 "
        "mismatched elements",
    "grpo_surrogate_loss":
        "the kernel accumulates a 4.2M-element loss in an uninitialised "
        "buffer, so its value depends on allocator state; legacy reported "
        "max|diff|=8.05 on the same tolerance",
    "int8_quantized_matmul":
        "2549/20000000 elements off by 1 after round(); legacy reported 189 "
        "mismatched elements on the same tolerance",
    "llama_transformer_block":
        "max|diff|~18-22 against tol=0.01; legacy reported 184/512000 "
        "mismatched elements",
    "lora_linear":
        "borderline fp32: max|diff|=1.5e-3 against atol=rtol=1e-3; legacy "
        "passes or fails depending on the random draw",
    "matmul":
        "fp32 K=4096 accumulation; max|diff|=0.11-347 against tol=0.01; "
        "legacy reported 184k-20.8M mismatched elements",
    "sliding_window_self_attention":
        "max|diff|~0.01-0.02 against tol=1e-3; legacy reported 103/640000 "
        "mismatched elements",
    "softmax_1D":
        "the kernel only fills part of the output: its values sum to ~0.009 "
        "instead of 1.0, and it emits NaN on some launches.  Legacy's "
        "assert torch.allclose(atol=1e-3) hid this because every softmax "
        "value is ~1e-5, far below atol",
    "swiglu_mlp_block":
        "max|diff|=0.86 against tol=1e-3; legacy reported 4189/4498500 "
        "mismatched elements",
}


def pybind_names() -> list[str]:
    src = (ROOT / "src" / "bindings.cu").read_text()
    return sorted(set(re.findall(r'm\.def\("([^"]+)"', src)))


def run_check(name: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, "-m", "harness", "check", name],
                          cwd=ROOT, capture_output=True, text=True)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # kernels print diagnostics; the verdict is the last PASS/FAIL/SKIP line
    verdict = [ln for ln in lines if ln.startswith(("PASS ", "FAIL ", "SKIP "))]
    if verdict:
        return proc.returncode, verdict[-1]
    if lines:
        return proc.returncode, lines[-1]
    return proc.returncode, ""


def perturb(value, expected, tol):
    """Move ``value`` a few tolerances away from ``expected``.

    A fixed +1% scale is not a vacuity test: specs with rtol=1e-2 (or values
    much smaller than atol) accept it.  The perturbation is therefore derived
    from the Spec's own tolerance -- 4x (atol + rtol*|expected|) -- which any
    correct comparison must reject.
    """
    import torch
    atol, rtol = tol
    if torch.is_tensor(value):
        if not value.is_floating_point():
            return value + 1
        exp = expected
        if torch.is_tensor(exp):
            if exp.shape != value.shape and exp.numel() == value.numel():
                exp = exp.reshape(value.shape)
            if exp.shape == value.shape:
                margin = (atol + rtol * exp.abs().to(torch.float64)) * 4.0
                return (value.to(torch.float64) + margin).to(value.dtype)
        return value * 1.5
    if isinstance(value, (tuple, list)):
        exp = expected if isinstance(expected, (tuple, list)) else [expected] * len(value)
        return type(value)(perturb(v, e, tol) for v, e in zip(value, exp))
    return value


def spot_check(names: list[str]) -> int:
    """Each sampled Spec must reject a 1% perturbation of its own output."""
    import torch
    from harness import cli

    failures = 0
    for name in names:
        if name in KNOWN_FAIL:
            print(f"spot {name}: SKIP (known kernel defect)")
            continue
        spec = registry.load_all()[name]
        if spec.reference is None:
            print(f"spot {name}: SKIP (no reference)")
            continue
        call, inputs = cli._prepare(spec)
        import copy
        ref_args = tuple(copy.deepcopy(a) for a in inputs[0])
        expected = spec.reference(*ref_args, **inputs[1])
        out = call()
        torch.cuda.synchronize()
        candidate = (tuple(inputs[0][i] for i in spec.inplace_indices)
                     if spec.inplace_indices else out)
        good = checker.compare_outputs(candidate, expected, spec.tol, spec.postprocess)
        if not good.ok:
            print(f"spot {name}: FAIL already at the clean comparison: {good.detail()}")
            failures += 1
            continue
        bad = checker.compare_outputs(perturb(candidate, expected, spec.tol),
                                      expected, spec.tol, spec.postprocess)
        if bad.ok:
            print(f"spot {name}: FAIL comparison accepts 4x-tolerance noise ({bad.detail()})")
            failures += 1
        else:
            print(f"spot {name}: ok rejects 4x-tolerance noise ({bad.detail()})")
    return failures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=int, default=0,
                    help="perturbation-check this many specs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--only", default=None, help="comma separated subset")
    args = ap.parse_args(argv)

    specs = registry.load_all()
    names = sorted(specs)
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]

    print(f"coverage: {len(pybind_names())} pybind names, {len(specs)} specs")
    missing = [n for n in pybind_names() if n not in specs]
    extra = [n for n in specs if n not in pybind_names()]
    for n in missing:
        tag = "benchmark-only" if n in BENCHMARK_ONLY else "MISSING SPEC"
        print(f"  {tag}: {n}")
    for n in extra:
        print(f"  spec without pybind entry: {n}")

    rows = []
    for i, name in enumerate(names, 1):
        rc, line = run_check(name)
        rows.append({"name": name, "rc": rc, "detail": line})
        print(f"[{i}/{len(names)}] {line[:160]}", flush=True)

    passed = [r for r in rows if r["rc"] == 0 and r["detail"].startswith("PASS")]
    skipped = [r for r in rows if r["rc"] == 0 and r["detail"].startswith("SKIP")]
    failed = [r for r in rows if r not in passed and r not in skipped]
    known = [r for r in failed if r["name"] in KNOWN_FAIL]
    failed = [r for r in failed if r["name"] not in KNOWN_FAIL]
    for r in known:
        print(f"  XFAIL {r['name']}: {KNOWN_FAIL[r['name']]}")
    print(f"\nverify: {len(passed)} PASS, {len(skipped)} SKIP, "
          f"{len(known)} XFAIL, {len(failed)} FAIL")

    spot_failures = 0
    if args.spot:
        spot_names = [n for n in names if n not in KNOWN_FAIL][:args.spot]
        spot_failures = spot_check(spot_names)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"rows": rows, "spot_failures": spot_failures}, indent=1))

    blocking_missing = [n for n in missing if n not in BENCHMARK_ONLY]
    ok = not failed and not blocking_missing and spot_failures == 0
    print("verify_specs:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
