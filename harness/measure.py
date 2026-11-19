"""CUDA-event timing with adaptive iteration count and robust statistics."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class Stats:
    n: int
    median: float
    trimmed_mean: float
    mean: float
    min: float
    max: float
    p10: float
    p90: float
    mad: float
    cv: float          # percent: pstdev / mean * 100

    def as_dict(self) -> dict:
        return asdict(self)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("cannot take a quantile of an empty sample")
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def summarize(samples: Sequence[float]) -> Stats:
    if not samples:
        raise ValueError("no samples")
    s = sorted(samples)
    n = len(s)
    mean = statistics.fmean(s)
    med = statistics.median(s)
    k = int(n * 0.1)
    core = s[k:n - k] if n - 2 * k >= 1 else s
    mad = statistics.median([abs(x - med) for x in s])
    cv = (statistics.pstdev(s) / mean * 100.0) if mean else 0.0
    return Stats(
        n=n, median=med, trimmed_mean=statistics.fmean(core), mean=mean,
        min=s[0], max=s[-1], p10=_quantile(s, 0.1), p90=_quantile(s, 0.9),
        mad=mad, cv=cv,
    )


def _event_time(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    del out
    return ms


def measure(fn: Callable[[], object], *, warmup: int = 5, target_ms: float = 200.0,
            min_iters: int = 20, max_iters: int = 200) -> Stats:
    """Warm up, then size the loop so each process runs >= target_ms of work."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    probe = _event_time(fn)
    iters = min_iters if probe <= 0 else int(round(target_ms / probe))
    iters = max(min_iters, min(max_iters, iters))

    return summarize([_event_time(fn) for _ in range(iters)])


def aggregate(per_proc: Sequence[Stats]) -> dict:
    """Cross-process aggregation: median-of-medians plus spread."""
    if not per_proc:
        raise ValueError("no process samples")
    meds = sorted(s.median for s in per_proc)
    agg_med = statistics.median(meds)
    spread = (meds[-1] - meds[0]) / agg_med * 100.0 if agg_med else 0.0
    worst = max(per_proc, key=lambda s: s.cv)
    return {
        "median": agg_med,
        "min": min(s.min for s in per_proc),
        "p90": max(s.p90 for s in per_proc),
        "spread_pct": spread,
        "cv": worst.cv,
        "n": int(statistics.median([s.n for s in per_proc])),
        "procs": len(per_proc),
    }
