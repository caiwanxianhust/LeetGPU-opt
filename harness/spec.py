"""Data contract for a benchmarkable kernel.

A Spec is the single source of truth for one kernel: how to build its inputs,
its torch reference, its tolerance, and the optional cost model used to turn
wall time into GFLOP/s or GB/s. test.py is intentionally NOT involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

Inputs = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass(frozen=True)
class Spec:
    name: str                      # canonical: must equal the pybind name
    make_inputs: Callable[[int], Inputs]
    test_fn: str | None = None     # legacy test.py function name, if any
    dtype: torch.dtype = torch.float32
    reference: Callable[..., Any] | None = None
    tol: tuple[float, float] = (1e-3, 1e-3)
    shape: Callable[[Inputs], str] | None = None
    flops: Callable[[Inputs], float] | None = None
    nbytes: Callable[[Inputs], float] | None = None
    variants: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"spec name must be a python identifier: {self.name!r}")
        if not callable(self.make_inputs):
            raise TypeError(f"{self.name}: make_inputs must be callable")
        atol, rtol = self.tol
        if atol < 0 or rtol < 0:
            raise ValueError(f"{self.name}: tolerance must be non-negative")

    @property
    def dtype_str(self) -> str:
        names = {
            torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16",
            torch.float64: "fp64", torch.int8: "int8", torch.int32: "int32",
        }
        return names.get(self.dtype, str(self.dtype).replace("torch.", ""))

    def shape_str(self, inputs: Inputs) -> str:
        return self.shape(inputs) if self.shape is not None else "-"
