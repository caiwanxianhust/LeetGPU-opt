"""Spec definitions: one per kernel you want to benchmark or check.

Adding a kernel = add one Spec here. The shapes live here and nowhere else;
test.py is legacy for the older kernels and is not consulted.
"""

from __future__ import annotations

import math

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

# --- flash_decoding_attention -------------------------------------------------
# Official perf shape from the kernel header: T=1, S=65536, H=32, D=128.
# Q is [T, H, D]; K/V are [S, H, D]; output is [T, H, D]. Inputs are drawn
# uniform in [-6, 6] as the problem statement specifies.

_T_FD, _S_FD, _H_FD, _D_FD = 1, 65536, 32, 128
_CHUNK_FD = 128


def _flash_decoding_inputs(seed: int):
    torch.manual_seed(seed)
    dev = "cuda"
    Q = torch.empty(_T_FD, _H_FD, _D_FD, device=dev, dtype=torch.float32).uniform_(-6.0, 6.0)
    K = torch.empty(_S_FD, _H_FD, _D_FD, device=dev, dtype=torch.float32).uniform_(-6.0, 6.0)
    V = torch.empty(_S_FD, _H_FD, _D_FD, device=dev, dtype=torch.float32).uniform_(-6.0, 6.0)
    return (Q, K, V), {}


def _flash_decoding_reference(Q, K, V):
    """Plain global softmax attention. Matmuls are chunked over S only to keep
    extra peak memory at ~8 MiB (K/V are 1 GiB each at the perf shape); the
    softmax itself is over the whole sequence, exactly like the kernel."""
    T, H, D = Q.shape
    S = K.shape[0]
    scores = torch.empty(T, H, S, device=Q.device, dtype=torch.float32)
    for s in range(0, S, 8192):
        scores[:, :, s:s + 8192] = torch.einsum("thd,chd->thc", Q, K[s:s + 8192])
    scores.mul_(1.0 / math.sqrt(D))
    w = torch.softmax(scores, dim=-1)
    out = torch.zeros(T, H, D, device=Q.device, dtype=torch.float32)
    for s in range(0, S, 8192):
        out += torch.einsum("thc,chd->thd", w[:, :, s:s + 8192], V[s:s + 8192])
    return out


register(Spec(
    name="flash_decoding_attention",
    dtype=torch.float32,
    make_inputs=_flash_decoding_inputs,
    reference=_flash_decoding_reference,
    tol=(1e-3, 1e-3),
    shape=lambda inputs: f"T{_T_FD}_S{_S_FD}_H{_H_FD}_D{_D_FD}",
    # QK^T plus PV, each 2*S*H*D MACs per query head.
    flops=lambda inputs: 4.0 * _T_FD * _S_FD * _H_FD * _D_FD,
    # Q, K, V, O plus the per-chunk O/lse workspace written then read back.
    nbytes=lambda inputs: 4.0 * (
        2 * _T_FD * _H_FD * _D_FD
        + 2 * _S_FD * _H_FD * _D_FD
        + 2 * _T_FD * _H_FD * ((_S_FD + _CHUNK_FD - 1) // _CHUNK_FD) * _D_FD
    ),
    notes="chunked flash decoding over S (CHUNK_SIZE=128) + LSE second pass; "
          "perf shape T=1, S=65536, H=32, D=128",
))
