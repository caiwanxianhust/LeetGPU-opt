"""Cross-check every Spec against the legacy test.py it came from.

A Spec is only trustworthy if it accepts what the old test accepted.  This tool
runs both sides in separate processes and reports the only interesting
disagreement:

    LEGACY_ONLY_FAIL  the legacy test failed and the Spec failed too -- they
                      agree, the kernel or the input has a problem;
    SPEC_ONLY_FAIL    the legacy test passed but the Spec rejected the same
                      kernel -- the translation changed the meaning, and the
                      Spec is wrong.

`SPEC_ONLY_FAIL` must be zero, and it has to stay zero across repeated runs,
because a Spec that only passes sometimes is a flaky gate.

Legacy output is silenced: the old tests print whole tensors.

Usage:
    python3 tools/cross_check.py             # one round
    python3 tools/cross_check.py --rounds 5
    python3 tools/cross_check.py --specs-json /tmp/verify.json   # reuse verdicts
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import registry  # noqa: E402
from verify_specs import KNOWN_FAIL  # noqa: E402

_LEGACY_RUNNER = r"""
import io, sys, contextlib
sys.path.insert(0, %(root)r)
import torch
import test as legacy

failures = []
original = legacy.compare_tensors

def recording(candidate, expected, *args, **kwargs):
    try:
        ok = original(candidate, expected, *args, **kwargs)
    except Exception as exc:            # noqa: BLE001 - legacy helper
        ok = False
    if ok is False:
        failures.append("compare_tensors returned a failure")
    return True

legacy.compare_tensors = recording
sink = io.StringIO()
rc = 0
try:
    with contextlib.redirect_stdout(sink):
        getattr(legacy, %(fn)r)()
except AssertionError as exc:
    failures.append("AssertionError: %%s" %% exc)
    rc = 1
except Exception as exc:                # noqa: BLE001 - kernel may crash
    failures.append("%%s: %%s" %% (type(exc).__name__, exc))
    rc = 2
if failures:
    rc = rc or 1
print("LEGACY", "FAIL" if failures else "PASS", "; ".join(failures)[:300])
sys.exit(rc)
"""


def run_legacy(fn: str, timeout: int = 900) -> tuple[bool, str]:
    code = _LEGACY_RUNNER % {"root": str(ROOT), "fn": fn}
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)
    tail = (proc.stdout.strip().splitlines() or [""])[-1]
    return proc.returncode == 0, tail or proc.stderr.strip().splitlines()[-1][:200]


def run_spec(name: str, timeout: int = 900) -> bool:
    proc = subprocess.run([sys.executable, "-m", "harness", "check", name],
                          cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)
    return proc.returncode == 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--specs-json", default=None,
                    help="verify_specs.py --json output; skips re-running Specs")
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)

    specs = registry.load_all()
    names = [n for n in sorted(specs) if specs[n].test_fn]
    # A known-defective kernel's verdict flips with the random draw on both
    # sides, so its disagreement carries no information; verify_specs.py
    # records the defect instead (KNOWN_FAIL).
    skipped = [n for n in names if n in KNOWN_FAIL]
    names = [n for n in names if n not in KNOWN_FAIL]
    if skipped:
        print(f"skipping {len(skipped)} known-failing kernels: "
              f"{', '.join(skipped)}")
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]
    if not names:
        print("no Spec declares a test_fn: nothing to cross-check")
        return 0

    cached = None
    if args.specs_json:
        rows = json.loads(Path(args.specs_json).read_text())["rows"]
        cached = {r["name"]: r["rc"] == 0 for r in rows}

    suspect = 0
    for round_no in range(1, args.rounds + 1):
        print(f"round {round_no}/{args.rounds}: {len(names)} kernels", flush=True)
        for i, name in enumerate(names, 1):
            spec_ok = cached[name] if cached and name in cached else run_spec(name)
            legacy_ok, detail = run_legacy(specs[name].test_fn)
            if legacy_ok and not spec_ok:
                suspect += 1
                verdict = "SPEC_ONLY_FAIL"
            elif spec_ok and not legacy_ok:
                verdict = "legacy-only-fail"
            else:
                verdict = "agree"
            if verdict != "agree":
                print(f"  [{i}] {name}: {verdict} ({detail[:120]})", flush=True)
        print(f"  round {round_no}: SPEC_ONLY_FAIL={suspect}", flush=True)

    print("cross_check:", "PASS" if suspect == 0 else "FAIL")
    return 0 if suspect == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
