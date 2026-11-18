# BENCH

RTX 4060 Laptop（sm_89, 8 GB）· CUDA 12.8 · torch 2.9.0+cu128

协议：`bench.py`，CUDA Event 计时，`seed=0`，warmup 5 / iter 20，取中位数；
**≥3 个独立进程**，跨进程取 median-of-medians，同时记录 min。
`vs 上一版` = `Δms (%)`，负值为变快；`vs PyTorch` = 同协议测得的 torch 参考实现的加速比（`--ref torch`）。

| 日期 | commit | kernel | shape | dtype | median ms | min ms | vs 上一版 | vs PyTorch | 改动 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | a78a189 | fused_layernorm_linear_gelu | 4096x4096x4096 | fp32 | 21.006 | 19.297 | 基线 | 待测 | naive |
| 2026-11-18 | 7aadc5a | fused_layernorm_linear_gelu | 4096x4096x4096 | fp32 | 19.767 | 19.201 | —（同代码漂移） | 1.1x | gate 框架自检，kernel 未改 |

## 噪声带

同版本重复测量得到的中位数区间，**低于此区间的差异不算改进**（门 B3 判无效）。

| kernel | 版本 commit | 单进程 median（6 次独立进程, ms） | 区间宽度 | 3 进程 median-of-medians 标准误 |
|---|---|---|---|---|
| fused_layernorm_linear_gelu | a78a189 | 20.698 / 20.850 / 21.006 / 21.222 / 22.151 / 22.750 | 2.052 ms (9.9%) | ≈ 0.43 ms (2.0%) |

结论：单进程 median 波动达 9.9%，2 进程 + 5% 门槛会误判；改取 3 进程 median-of-medians 后标准误约 2%，
故门 B3 的 **8% 门槛 ≈ 4σ**，可判。若某 kernel 的噪声带更宽，以表中实测值为准，不得沿用 8%。

**跨 session 漂移（重要）**：同一份未改动的二进制，`a78a189` 时段量到 median 21.0 ms，
同日在 gate 自检中量到 19.6 ms（≈7% 差异，来自 GPU 时钟/温度/宿主负载）。
因此**跨 session 比较无效**——B3 判定前必须在同一 session 内先重新冻结基线（工作流 B 步骤 1），
不要把历史行的数字直接当今天的基线。gate 会把这种"假改进"判 FAIL，属预期行为。

> 附带观察：首次进程偶发单次 44.8–63.3 ms 尖峰（笔记本/WSL2 宿主的电源与调度抖动），
> 中位数对其稳健；跨进程仍建议丢弃第 1 个进程的中位数或至少跑 3 个。
