# BENCH

RTX 4060 Laptop（sm_89, 8 GB, 24 SM）· CUDA 12.8 · torch 2.9.0+cu128

## 协议（由 harness 执行，不再手抄）

- `python3 -m harness run <kernel> --procs 3 --ref torch`：固定 `seed=0`，warmup 5，
  自适应迭代（目标每进程 ≥200 ms，夹在 20–200 iter），每进程取中位数，跨进程取 **median-of-medians**。
- 结果写入 `BENCH.json`，含 `so_sha256`（钉住被测二进制）、git commit、GPU、torch/cuda 版本。
- 判定（`harness compare`）：`gain = 1 - cand/base`，接受条件
  **`gain ≥ max(8%, 3×噪声SE)` 且 `cand.p90 ≤ base.p90 × 1.02`**。
  噪声 SE：多个进程时用 `spread/sqrt(procs)`，单进程时退化为 `cv/sqrt(n)`。
  实测本机 3 进程 spread ≈ 0.9% → 3×SE ≈ 1.6%，故 8% 下限主导。
- 下表由 `python3 -m harness report` 从 `BENCH.json` 生成，**不要手改标记区间内的行**。

## 记录

commit 列带 `*` 表示记录时工作区有未提交改动（`git.dirty=true`）：该行数字对应那棵临时工作树，
提交后应重跑一次确认。行由 `harness report` 生成，**不要手改标记区间内内容**。

<!-- BENCH:TABLE:START -->
| 日期 | commit | kernel | shape | dtype | median ms | min ms | vs 上一版 | vs PyTorch | 改动 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | 1d61067* | fused_ln_linear_gelu | 4096x4096x4096 | fp32 | 19.597 | 19.274 | 基线 | 1.05x | gate |
<!-- BENCH:TABLE:END -->

## 历史（旧 `bench.py` 口径，仅存档不参与判定）

| 日期 | commit | kernel | shape | dtype | median ms | min ms | vs 上一版 | vs PyTorch | 改动 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-18 | a78a189 | fused_layernorm_linear_gelu | 4096x4096x4096 | fp32 | 21.006 | 19.297 | 基线 | 待测 | naive |
| 2026-11-18 | 7aadc5a | fused_layernorm_linear_gelu | 4096x4096x4096 | fp32 | 19.767 | 19.201 | —（同代码漂移） | 1.1x | gate 框架自检，kernel 未改 |

## 噪声带

同版本重复测量得到的中位数区间，**低于此区间的差异不算改进**。

| 观测 | 结果 |
|---|---|
| 单进程 median，6 次独立进程（2026-11-18） | 20.698 / 20.850 / 21.006 / 21.222 / 22.151 / 22.750 ms（跨度 9.9%） |
| 3 进程 median-of-medians，`harness run --procs 3` | 19.597 ms，跨进程 spread 0.6%，单进程 CV 3.4%；torch 参考 20.567 ms（1.05x） |
| 同一二进制跨 session | 21.0 ms → 19.6 ms（≈7% 漂移，GPU 时钟/温度/宿主负载） |

结论：
1. 单进程 + 5% 门槛会误判，已废弃；现用 3 进程 + 8% 下限。
2. **跨 session 比较无效**——判定前必须在同一 session 内重新冻结基线（工作流 B 步骤 1）。
3. 首次进程偶发 44.8–63.3 ms 单次尖峰（WSL2/笔记本电源管理）；中位数对尖峰稳健，
   但跨进程仍建议 ≥3 个，gate 在全部门 PASS 后才记账。
