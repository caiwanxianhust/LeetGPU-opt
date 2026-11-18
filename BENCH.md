# BENCH

RTX 4060 Laptop（sm_89, 8 GB）· CUDA 12.8 · torch 2.9.0+cu128 · fp32
协议：`bench.py`，CUDA Event 计时，warmup 5 / iter 20，取中位数；**≥2 个独立进程**。
噪声参考：median 波动约 ±5%（`fused_layernorm_linear_gelu` 4096³ 实测 20.7–22.8 ms）。

| 日期 | commit | kernel | shape | median ms | min ms | vs 上一版 | vs PyTorch | 改动 |
|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | a78a189 | fused_layernorm_linear_gelu | 4096³ | 21.0 | 19.3 | 基线 | 待测 | naive |