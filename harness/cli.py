"""Command line for the leetgpu benchmark harness.

    python3 -m harness run <kernel> [--procs N] [--ncu] [--ref torch] [--json PATH] [--markdown]
    python3 -m harness check <kernel>
    python3 -m harness compare <base.json> <cand.json> [--min-gain PCT]
    python3 -m harness report [--json BENCH.json] [--md BENCH.md]
    python3 -m harness suite
    python3 -m harness selftest

stdout stays machine-readable for `run` (one CSV line) unless --json is used;
all diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys

import torch

from . import checker, judge, measure, registry, report, store

SEED = 0
WARMUP = 5
TARGET_MS = 200.0
MIN_ITERS = 20
MAX_ITERS = 200


def _require_cuda() -> bool:
    if torch.cuda.is_available():
        return True
    print(
        "CUDA unavailable. Under the workspace-write sandbox this shows up as "
        "`CUDA error 304: OS call failed` while nvidia-smi works; rerun with "
        "Full access (AGENTS.md rule 4).",
        file=sys.stderr,
    )
    return False


def _resolve(name: str):
    try:
        return registry.resolve(name)
    except KeyError:
        print(f"unknown kernel: {name}\nsupported: {', '.join(registry.names())}",
              file=sys.stderr)
        return None, None


def _prepare(spec):
    import leetgpu
    fn = getattr(leetgpu, spec.name, None)
    if fn is None:
        raise SystemExit(f"leetgpu has no attribute {spec.name!r}; rebuild the extension")
    torch.cuda.synchronize()
    torch.manual_seed(SEED)
    inputs = spec.make_inputs(SEED)
    call = lambda: fn(*inputs[0], **inputs[1])  # noqa: E731
    return call, inputs


def _measure_once(spec, ncu: bool) -> measure.Stats:
    call, _ = _prepare(spec)
    if ncu:
        for _ in range(WARMUP):
            call()
        torch.cuda.synchronize()
        return measure.summarize([measure._event_time(call)])
    return measure.measure(call, warmup=WARMUP, target_ms=TARGET_MS,
                           min_iters=MIN_ITERS, max_iters=MAX_ITERS)


def _protocol(ncu: bool) -> dict:
    return {
        "seed": SEED,
        "warmup": WARMUP,
        "ncu": ncu,
        "target_ms": 0 if ncu else TARGET_MS,
        "min_iters": 1 if ncu else MIN_ITERS,
        "max_iters": 1 if ncu else MAX_ITERS,
    }


def _md_row(record: dict) -> str:
    agg = record["agg"]
    ref = record.get("ref") or {}
    speedup = f"{float(ref['speedup']):.2f}x" if ref.get("speedup") else "待测"
    return (f"| {record['ts'][:10]} | {record['git']['commit']} | {record['kernel']} "
            f"| {record['shape']} | {record['dtype']} | {float(agg['median']):.3f} "
            f"| {float(agg['min']):.3f} | 基线 | {speedup} | {record.get('note') or '—'} |")


def _emit(args, record: dict) -> int:
    agg = record["agg"]
    print(f"# agg: procs={agg['procs']} n={agg.get('n', '?')} "
          f"median={agg['median']:.3f}ms min={agg['min']:.3f}ms p90={agg['p90']:.3f}ms "
          f"spread={agg['spread_pct']:.1f}% worst_cv={agg['cv']:.2f}%", file=sys.stderr)
    if args.json == "-":
        print(json.dumps(record, ensure_ascii=False))
    elif args.markdown:
        print(_md_row(record))
    else:
        print(f"{record['kernel']},{record['shape']},{record['dtype']},"
              f"{float(agg['median']):.3f},{float(agg['min']):.3f}")
    if args.json and args.json != "-":
        store.append_record(args.json, record)
        print(f"# appended -> {args.json}", file=sys.stderr)
    return 0


def _torch_ref(spec, median_ms: float) -> dict | None:
    if spec.reference is None:
        return None
    _, inputs = _prepare(spec)
    stats = measure.measure(lambda: spec.reference(*inputs[0], **inputs[1]),
                            warmup=WARMUP, target_ms=TARGET_MS,
                            min_iters=MIN_ITERS, max_iters=MAX_ITERS)
    speedup = (stats.median / median_ms) if median_ms else None
    print(f"# ref(torch): median={stats.median:.3f}ms min={stats.min:.3f}ms "
          f"speedup={speedup:.2f}x" if speedup else "", file=sys.stderr)
    return {"median": stats.median, "min": stats.min, "speedup": speedup}


def cmd_run(args) -> int:
    if args.list:
        for name in registry.names():
            spec = registry.load_all()[name]
            print(f"{name}\t(dtype={spec.dtype_str}, test_fn={spec.test_fn})", file=sys.stderr)
        return 0
    if not args.kernel:
        print("usage: python3 -m harness run <kernel>", file=sys.stderr)
        return 2

    canonical, spec = _resolve(args.kernel)
    if canonical is None:
        return 2
    if args.print_test_fn:
        if not spec.test_fn:
            print(f"{canonical} has no test_fn registered", file=sys.stderr)
            return 2
        print(spec.test_fn)
        return 0
    if not _require_cuda():
        return 3

    if args.procs > 1:
        return _run_procs(args, canonical, spec)

    stats = _measure_once(spec, args.ncu)
    _, inputs = _prepare(spec)
    agg = measure.aggregate([stats])
    ref = _torch_ref(spec, agg["median"]) if args.ref == "torch" else None
    record = store.build_record(kernel=canonical, shape=spec.shape_str(inputs),
                                dtype=spec.dtype_str, protocol=_protocol(args.ncu),
                                procs=[stats], agg=agg, ref=ref, note=args.note)
    return _emit(args, record)


def _run_procs(args, canonical: str, spec) -> int:
    """Fan out to N real OS processes; aggregate median-of-medians."""
    stats: list[measure.Stats] = []
    child = None
    for i in range(args.procs):
        cmd = [sys.executable, "-m", "harness", "run", canonical, "--procs", "1",
               "--json", "-", "--_child"]
        if args.ncu:
            cmd.append("--ncu")
        proc = subprocess.run(cmd, cwd=store.repo_root(), capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode or 1
        child = json.loads(proc.stdout)
        stats.append(measure.Stats(**child["procs"][0]))
        print(f"# proc {i + 1}/{args.procs}: median={stats[-1].median:.3f}ms "
              f"min={stats[-1].min:.3f}ms cv={stats[-1].cv:.2f}% n={stats[-1].n}",
              file=sys.stderr)

    agg = measure.aggregate(stats)
    ref = _torch_ref(spec, agg["median"]) if args.ref == "torch" else None
    record = store.build_record(kernel=canonical, shape=child["shape"],
                                dtype=child["dtype"],
                                protocol={**_protocol(args.ncu), "procs": args.procs},
                                procs=stats, agg=agg, ref=ref, note=args.note)
    return _emit(args, record)


def cmd_check(args) -> int:
    canonical, spec = _resolve(args.kernel)
    if canonical is None:
        return 2
    if not _require_cuda():
        return 3
    if spec.reference is None:
        # Benchmark-only spec: there is a kernel but no trustworthy reference
        # (see README "kernels without a Spec").  SKIP is not a failure.
        print(f"SKIP {canonical}: no reference registered (benchmark-only spec)",
              file=sys.stderr)
        return 0

    call, inputs = _prepare(spec)
    # The reference must see the inputs as they were *before* the kernel ran:
    # in-place kernels overwrite their arguments.  Give the reference its own
    # copies so neither side can disturb the other.
    ref_args = tuple(copy.deepcopy(a) for a in inputs[0])
    expected = spec.reference(*ref_args, **inputs[1])
    out = call()
    torch.cuda.synchronize()

    if spec.inplace_indices:
        # The kernel mutates arguments; compare the mutated arguments, not the
        # return value (which legacy tests usually ignored or returned as None).
        candidate = tuple(inputs[0][i] for i in spec.inplace_indices)
    else:
        candidate = out

    atol, rtol = spec.tol
    result = checker.compare_outputs(candidate, expected, spec.tol, spec.postprocess)
    if not result.ok:
        print(f"FAIL {canonical}: {result.detail()} (atol={atol} rtol={rtol})",
              file=sys.stderr)
        return 1
    print(f"PASS {canonical}: {result.detail()} within atol={atol} rtol={rtol}",
          file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    base_records = store.load_records(args.base)
    cand_records = store.load_records(args.cand)
    if not base_records or not cand_records:
        print("empty record file", file=sys.stderr)
        return 2
    base, cand = base_records[-1], cand_records[-1]
    print(f"base: {base['kernel']} commit={base['git']['commit']} "
          f"median={float(base['agg']['median']):.3f}ms so={base.get('so_sha256')}")
    print(f"cand: {cand['kernel']} commit={cand['git']['commit']} "
          f"median={float(cand['agg']['median']):.3f}ms so={cand.get('so_sha256')}")
    verdict = judge.decide(base["agg"], cand["agg"], min_gain_pct=args.min_gain)
    print(verdict.line())
    return 0 if verdict.accept else 1


def cmd_report(args) -> int:
    records = store.load_records(args.json)
    if not records:
        print(f"no records in {args.json}", file=sys.stderr)
        return 2
    report.update_markdown(args.md, records)
    print(f"updated {args.md} from {len(records)} record(s)")
    return 0


def cmd_suite(args) -> int:
    if not _require_cuda():
        return 3
    failures = 0
    for name in registry.names():
        stats = _measure_once(registry.load_all()[name], False)
        _, inputs = _prepare(registry.load_all()[name])
        agg = measure.aggregate([stats])
        print(f"{name},{registry.load_all()[name].shape_str(inputs)},"
              f"{registry.load_all()[name].dtype_str},"
              f"{float(agg['median']):.3f},{float(agg['min']):.3f}")
    return 1 if failures else 0


def cmd_selftest(args) -> int:
    cases = [
        ("10% faster on a tight measurement accepted",
         {"median": 100.0, "spread_pct": 0.9, "procs": 3, "p90": 101.0},
         {"median": 88.0, "spread_pct": 0.9, "procs": 3, "p90": 89.0}, True),
        ("2% is inside the floor",
         {"median": 100.0, "spread_pct": 0.9, "procs": 3, "p90": 101.0},
         {"median": 98.0, "spread_pct": 0.9, "procs": 3, "p90": 99.0}, False),
        ("unstable 2-process run raises the bar",
         {"median": 100.0, "spread_pct": 15.0, "procs": 2, "p90": 101.0},
         {"median": 90.0, "spread_pct": 15.0, "procs": 2, "p90": 95.0}, False),
        ("single-process high CV falls back to CV/sqrt(n)",
         {"median": 100.0, "cv": 20.0, "n": 20, "procs": 1, "p90": 101.0},
         {"median": 90.0, "cv": 20.0, "n": 20, "procs": 1, "p90": 95.0}, False),
        ("tail regression rejected despite median gain",
         {"median": 100.0, "spread_pct": 0.9, "procs": 3, "p90": 101.0},
         {"median": 85.0, "spread_pct": 0.9, "procs": 3, "p90": 110.0}, False),
    ]
    failures = 0
    for label, base, cand, expected in cases:
        verdict = judge.decide(base, cand)
        good = verdict.accept == expected
        failures += 0 if good else 1
        print(f"{'ok  ' if good else 'FAIL'} {label}: {verdict.line()}")
    failures += _check_selftest()
    print(f"selftest: {'PASS' if not failures else f'{failures} failed'}")
    return 0 if not failures else 1


def _check_selftest() -> int:
    """Unit-test the comparison semantics of `harness check` (no GPU needed)."""
    base = (torch.arange(6, dtype=torch.float32).reshape(2, 3) / 7.0)
    ints = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    nan = torch.tensor([float("nan"), 1.0])

    def cmp(cand, exp, tol=(1e-3, 1e-3), pp=None):
        return checker.compare_outputs(cand, exp, tol, pp)

    cases = [
        ("float inside tolerance accepted", True, cmp(base, base + 1e-5)),
        ("float outside tolerance rejected", False, cmp(base, base + 1e-2)),
        ("tuple of two outputs accepted", True, cmp((base, ints), (base, ints))),
        ("tuple arity mismatch rejected", False, cmp((base,), (base, ints))),
        ("integer output compares exactly", True, cmp(ints, ints.clone())),
        ("integer off-by-one rejected", False,
         cmp(ints, torch.tensor([[1, 2], [3, 5]], dtype=torch.int32))),
        ("shape mismatch rejected", False, cmp(base.reshape(3, 2), base)),
        ("0-dim vs 1-element accepted", True,
         cmp(torch.tensor([3.0]), torch.tensor(3.0))),
        ("1-element vs 2-element rejected", False,
         cmp(torch.tensor([3.0]), torch.tensor([3.0, 4.0]))),
        ("dtype mismatch rejected", False, cmp(base, base.to(torch.float16))),
        ("postprocess is applied to the candidate only", True,
         cmp(base.reshape(6), base, pp=lambda t: t.view(2, 3))),
        ("NaN is not silently equal", False, cmp(nan, torch.tensor([0.0, 1.0]))),
    ]
    failures = 0
    for label, expected, result in cases:
        good = result.ok == expected
        failures += 0 if good else 1
        print(f"{'ok  ' if good else 'FAIL'} {label}: {result.detail()}")
    print(f"check-selftest: {'PASS' if not failures else f'{failures} failed'}")
    return failures


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m harness", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="measure one kernel")
    run.add_argument("kernel", nargs="?")
    run.add_argument("--procs", type=int, default=1, help="independent OS processes (default 1)")
    run.add_argument("--ncu", action="store_true", help="warmup + one launch, for ncu -s/-c")
    run.add_argument("--ref", choices=["torch"], help="also time the torch reference")
    run.add_argument("--json", default=None, help="append record here ('-' = stdout)")
    run.add_argument("--markdown", action="store_true", help="print a BENCH.md row")
    run.add_argument("--note", default="", help="free-form note stored in the record")
    run.add_argument("--list", action="store_true")
    run.add_argument("--print-test-fn", action="store_true")
    run.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)

    check = sub.add_parser("check", help="correctness against the spec reference")
    check.add_argument("kernel")

    cmp_ = sub.add_parser("compare", help="judge a candidate record against a baseline record")
    cmp_.add_argument("base")
    cmp_.add_argument("cand")
    cmp_.add_argument("--min-gain", type=float, default=8.0)

    rep = sub.add_parser("report", help="rebuild the BENCH.md table from BENCH.json")
    rep.add_argument("--json", default="BENCH.json")
    rep.add_argument("--md", default="BENCH.md")

    sub.add_parser("suite", help="measure every registered spec")
    sub.add_parser("selftest", help="unit-test the decision logic (no GPU needed)")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    return {
        "run": cmd_run,
        "check": cmd_check,
        "compare": cmd_compare,
        "report": cmd_report,
        "suite": cmd_suite,
        "selftest": cmd_selftest,
    }[args.cmd](args)
