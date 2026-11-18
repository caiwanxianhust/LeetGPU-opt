"""Spec definitions: one per kernel you want to benchmark or check.

Adding a kernel = add one Spec here. The shapes live here and nowhere else;
test.py is legacy for the older kernels and is not consulted.
"""

from __future__ import annotations

import torch

from .registry import register
from .spec import Spec

_MNK = 4096


def _fused_inputs(seed: int):
    torch.manual_seed(seed)
    x = torch.randn(_MNK, _MNK, device="cuda", dtype=torch.float32)
    gamma = torch.randn(_MNK, device="cuda", dtype=torch.float32)
    beta = torch.randn(_MNK, device="cuda", dtype=torch.float32)
    W = torch.randn(_MNK, _MNK, device="cuda", dtype=torch.float32)
    bias = torch.randn(_MNK, device="cuda", dtype=torch.float32)
    return (x, gamma, beta, W, bias, 1e-5), {}


def _fused_reference(x, gamma, beta, W, bias, eps):
    mean = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    x_norm = (x - mean) * gamma / torch.sqrt(var + eps) + beta
    return torch.nn.functional.gelu(x_norm @ W + bias)


def _fused_shape(inputs) -> str:
    x = inputs[0][0]
    W = inputs[0][3]
    return f"{x.shape[0]}x{x.shape[1]}x{W.shape[1]}"


register(Spec(
    name="fused_ln_linear_gelu",
    test_fn="fused_layernorm_linear_gelu",   # legacy name in test.py
    dtype=torch.float32,
    make_inputs=_fused_inputs,
    reference=_fused_reference,
    tol=(1e-3, 1e-3),
    shape=_fused_shape,
    # matmul dominates: 2*M*N*K; LN and GELU epilogue ignored.
    flops=lambda inputs: 2.0 * _MNK ** 3,
    # read x + W, write y (fp32); the small vectors are noise.
    nbytes=lambda inputs: 3.0 * _MNK * _MNK * 4,
    notes="LayerNorm + Linear + GELU fused into one kernel",
))
