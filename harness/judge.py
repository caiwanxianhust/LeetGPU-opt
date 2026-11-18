"""Accept/reject decision for a candidate measurement vs a frozen baseline.

Pure functions only, so this is unit-testable without a GPU (`harness selftest`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    accept: bool
    gain_pct: float
    threshold_pct: float
    reason: str

    def line(self) -> str:
        tag = "ACCEPT" if self.accept else "REJECT"
        return f"{tag}: {self.reason}"


def noise_se_pct(agg: dict) -> float:
    """Standard error of the reported median, in percent.

    Across-process spread is the honest run-to-run noise (per-iteration jitter is
    already removed by taking the median inside each process). With only one
    process we fall back to the within-process CV over n samples.
    """
    procs = int(agg.get("procs", 1) or 1)
    if procs >= 2 and "spread_pct" in agg:
        return float(agg["spread_pct"]) / math.sqrt(procs)
    n = int(agg.get("n", 20) or 20)
    return float(agg.get("cv", 0.0)) / math.sqrt(max(1, n))


def decide(base: dict, cand: dict, *, min_gain_pct: float = 8.0,
           tail_slack: float = 0.02) -> Verdict:
    """Accept only if the gain clears both the floor and 3x the noise SE."""
    b_med = float(base.get("median", 0.0))
    c_med = float(cand.get("median", 0.0))
    if b_med <= 0 or c_med <= 0:
        return Verdict(False, 0.0, min_gain_pct, "median must be positive")

    gain = (b_med - c_med) / b_med * 100.0
    se = max(noise_se_pct(base), noise_se_pct(cand))
    threshold = max(min_gain_pct, 3.0 * se)

    if gain < threshold:
        return Verdict(
            False, gain, threshold,
            f"gain {gain:+.1f}% < threshold {threshold:.1f}% "
            f"(floor {min_gain_pct:.0f}%, 3xSE {3.0 * se:.1f}%)",
        )

    b_p90 = float(base.get("p90", b_med))
    c_p90 = float(cand.get("p90", c_med))
    if c_p90 > b_p90 * (1.0 + tail_slack):
        return Verdict(
            False, gain, threshold,
            f"median gain {gain:+.1f}% but p90 regressed "
            f"{b_p90:.3f} -> {c_p90:.3f} ms (> {tail_slack * 100:.0f}%)",
        )

    return Verdict(
        True, gain, threshold,
        f"gain {gain:+.1f}% >= threshold {threshold:.1f}%, p90 ok",
    )
