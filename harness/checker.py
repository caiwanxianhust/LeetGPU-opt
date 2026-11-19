"""Output comparison for `harness check` (pure: no CUDA, no measurement).

Legacy test.py compared kernel output against a torch reference in several
mutually inconsistent ways, and the most common one -- a bare
``compare_tensors(out, ref)`` whose return value is dropped -- reported a
mismatch only as printed text and still exited 0.  This module is the single,
unit-tested definition of "equal enough" used by `harness check`:

* a kernel may return a tensor, a python scalar, or a tuple/list of those
  (some kernels return several tensors, e.g. fused AdamW returns nothing but
  updates params/m/v in place -- see `Spec.inplace_indices`);
* floating leaves compare with `torch.allclose(atol, rtol)`;
* integer / unsigned / bool leaves compare exactly (`torch.equal`): no
  approximate comparison is ever applied to indices or counts;
* shape and dtype must match exactly;
* `postprocess`, when given, is applied to each candidate leaf only -- the
  legacy code reshaped the kernel output (e.g. `out.view(N, N)`) and left the
  reference untouched, so applying it to both sides would be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

Tol = tuple[float, float]


@dataclass(frozen=True)
class Result:
    """Outcome of comparing one kernel call against its reference."""

    ok: bool
    n_outputs: int
    messages: list[str] = field(default_factory=list)
    max_diff: float | None = None

    def detail(self) -> str:
        head = f"{self.n_outputs} output{'s' if self.n_outputs != 1 else ''}"
        if not self.messages:
            return head
        return f"{head}, " + "; ".join(self.messages)


def flatten(value: Any) -> list[Any]:
    """Expand tuples/lists one level deep; everything else is a single leaf."""
    if isinstance(value, (tuple, list)):
        out: list[Any] = []
        for item in value:
            out.extend(flatten(item))
        return out
    return [value]


def _as_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (bool, int, float)):
        return torch.as_tensor(value)
    return None


def _compare_leaf(cand: torch.Tensor, exp: torch.Tensor,
                  tol: Tol) -> tuple[bool, str, float | None]:
    if cand.shape != exp.shape:
        # A 0-dim result and a 1-element result are the same scalar.  Legacy
        # tests for counts/losses compared with `.item()` and never pinned
        # the shape, so accept that pair; every other mismatch is real.
        if cand.numel() == 1 and exp.numel() == 1 and (cand.dim() == 0 or exp.dim() == 0):
            cand = cand.reshape(())
            exp = exp.reshape(())
        else:
            return False, f"shape {tuple(cand.shape)} != reference {tuple(exp.shape)}", None
    if cand.dtype != exp.dtype:
        # A reference written in python tends to be int64 where the kernel
        # returns int32 (counts, indices).  Integers are compared exactly
        # either way, so widen both sides instead of failing on the label.
        integral = not (cand.is_floating_point() or exp.is_floating_point()
                        or cand.is_complex() or exp.is_complex())
        if not integral:
            return False, f"dtype {cand.dtype} != reference {exp.dtype}", None
        cand = cand.to(torch.int64)
        exp = exp.to(torch.int64)
    if cand.is_floating_point() or cand.is_complex():
        atol, rtol = tol
        if cand.numel() == 0:
            return True, "max|diff|=0", 0.0
        a = cand.to(torch.float64)
        b = exp.to(torch.float64)
        max_diff = float((a - b).abs().max().item())
        ok = bool(torch.allclose(cand, exp, atol=atol, rtol=rtol, equal_nan=False))
        return ok, f"max|diff|={max_diff:.6g}", max_diff
    mismatched = int((cand != exp).sum().item())
    if mismatched:
        return False, f"{mismatched}/{cand.numel()} elements differ", None
    return True, "exact match", None


def compare_outputs(candidate: Any, expected: Any, tol: Tol,
                    postprocess: Callable[[Any], Any] | None = None) -> Result:
    """Compare a kernel result against the reference result of the same Spec."""
    cand_leaves = flatten(candidate)
    exp_leaves = flatten(expected)
    if postprocess is not None:
        cand_leaves = [postprocess(leaf) for leaf in cand_leaves]
    n = len(cand_leaves)
    if n != len(exp_leaves):
        return Result(False, n, [
            f"reference returned {len(exp_leaves)} output(s), kernel returned {n}"])

    diffs: list[float] = []
    failures: list[str] = []
    for i, (raw_cand, raw_exp) in enumerate(zip(cand_leaves, exp_leaves)):
        cand = _as_tensor(raw_cand)
        exp = _as_tensor(raw_exp)
        if cand is None:
            failures.append(f"output[{i}]: kernel returned {type(raw_cand).__name__}, not a tensor")
            continue
        if exp is None:
            failures.append(f"output[{i}]: reference returned {type(raw_exp).__name__}, not a tensor")
            continue
        ok, msg, max_diff = _compare_leaf(cand, exp, tol)
        if max_diff is not None:
            diffs.append(max_diff)
        if not ok:
            failures.append(f"output[{i}]: {msg}" if n > 1 else msg)

    if failures:
        return Result(False, n, failures, max(diffs) if diffs else None)
    if diffs:
        return Result(True, n, [f"max|diff|={max(diffs):.6g}"], max(diffs))
    return Result(True, n, ["exact match"], None)
