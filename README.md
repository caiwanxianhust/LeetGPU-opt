# LeetGPU · CUDA kernel 练习仓库

单卡（RTX 4060 Laptop, sm_89）CUDA kernel 练习与优化仓库。一题一个 `.cu`，统一编译成一个 Python 扩展
`leetgpu`；仓库自带一套**可复现的测量、判定、记账流程**，让"新增 kernel"和"优化 kernel"都有机器可判的门槛，
而不是靠肉眼看打印出来的毫秒数。

> 写给未来的自己 / 协作者：**先读本文，再读 [AGENTS.md](AGENTS.md)**。
> 本文解释"东西都在哪、各自干什么"，AGENTS.md 规定"动手时必须按什么顺序、过哪些门"。

---

## 1. 快速开始

```bash
cd /home/houlai/resumeProject/hpc/leetgpu

# 环境（只看一次就够）
nvcc --version                 # CUDA 12.8
nvidia-smi --query-gpu=name --format=csv,noheader

# 构建（内存自适应并发，默认按可用内存算，见 §4.1）
python3 setup.py build_ext --inplace

# 正确性（对标 torch 参考，机器判定）
python3 -m harness check fused_ln_linear_gelu

# 测一次（3 个独立进程，输出 CSV + 诊断）
python3 -m harness run fused_ln_linear_gelu --procs 3 --ref torch

# 全流程门禁：构建 → 正确性 → 性能 → 卫生 → 记账
bash tools/gate.sh fused_ln_linear_gelu
```

⚠️ **构建 / 测试 / 跑分必须在 Full access 权限下跑**。workspace-write 沙箱里 CUDA 会报
`CUDA error 304: OS call failed`，而 `nvidia-smi` 一切正常——这不是环境坏了，是沙箱挡住了 `/dev/dxg`。

---

## 2. 目录总览

```
leetgpu/
├── setup.py              # 唯一构建入口（内存感知 MAX_JOBS）
├── harness/              # ★ 评估框架：规格、测量、判定、记账（836 行）
├── bench.py              # 兼容壳 → harness run
├── tools/gate.sh         # 门禁执行器（B0–B5）
├── BENCH.json            # 机器可读测量历史
├── BENCH.md              # 人读账本（表格自动生成 + 手写散文）
├── AGENTS.md             # 给 agent 的工作流与家规
├── src/                  # 97 个 *.cu（96 个 kernel + bindings.cu）+ 2 个头文件
├── test.py               # 【legacy】存量 kernel 的测试脚本（4386 行）
├── profile/              # 【legacy】4 个 ncu 剖析脚本 + 6 个 .ncu-rep
├── build/                # 构建中间产物（gitignore）
├── leetgpu.cpython-310-x86_64-linux-gnu.so   # 构建产物（gitignore）
└── .vscode/              # 编辑器 *.cu→cpp 语言关联
```

数据流一句话：**Spec 造输入 → 直接调 pybind 入口 → CUDA Event 计时 → 统计 → 判定 → 写 `BENCH.json` → 渲染 `BENCH.md`**。

---

## 3. 逐文件说明

### 3.1 根目录

| 文件 | 行数 | 作用 |
|---|---|---|
| [setup.py](setup.py) | 153 | **唯一构建入口**。`_adaptive_jobs()` 读 `/proc/meminfo` 决定并发数；`glob.glob("src/*.cu")` 全量编译；硬编码 CUTLASS include、`-arch=sm_89 -std=c++17`；产物 `--inplace` 落到仓库根。改 GPU 架构只能改这里。 |
| [bench.py](bench.py) | 14 | 兼容壳。`python3 bench.py <k> [flags]` 等价于 `python3 -m harness run <k> [flags]`，保留旧习惯与旧脚本。 |
| [test.py](test.py) | 4386 | **legacy** 自测脚本。93 个测试函数（每 kernel 一个），两种判据并存：早期 28 处 `assert torch.allclose(...)`（共 31 条 assert）可靠；后期 71 处裸 `compare_tensors(...)` 调用丢弃返回值，**比较失败也 exit 0**。`__main__` 里只放开一个函数。新 kernel 不再往这里加。 |
| [AGENTS.md](AGENTS.md) | 185 | 给 agent 的契约：工作流 A（新增）/ B（优化）、门表 B0–B5、5 条家规。**动手前必读**。 |
| [BENCH.md](BENCH.md) | 49 | 人读账本。分四段：协议、生成表格（标记区间内自动覆盖）、历史存档（旧口径，不参与判定）、噪声带结论。 |
| [BENCH.json](BENCH.json) | — | 机器可读历史，每条含 `so_sha256`、git commit/dirty、GPU、torch/cuda 版本、每进程统计、聚合值、torch 参考。目前 2 条。 |
| [.gitignore](.gitignore) | 6 | 忽略 `build/`、`*.ncu-rep`、根目录 `.so`、`__pycache__/`、`*.py[cod]`。**新增产物类型必须同步补这里**（家规 1）。 |
| `.vscode/` | — | 仅把 `*.cu`/`*.cuh` 关联成 C++，便于跳转，无构建作用。 |

### 3.2 `harness/` —— 评估框架（新）

依赖方向单一：`cli → {registry → spec ← kernels, measure, judge, store, report}`。

| 文件 | 行数 | 作用 |
|---|---|---|
| [harness/__init__.py](harness/__init__.py) | 14 | 包说明与模块清单。 |
| [harness/__main__.py](harness/__main__.py) | 5 | 让 `python3 -m harness` 可用，转调 `cli.main()`。 |
| [harness/spec.py](harness/spec.py) | 49 | **数据契约**。`Spec` 冻结 dataclass：`name`（必须等于 pybind 名）、`test_fn`（legacy 名，可选）、`dtype`、`make_inputs(seed)`、`reference`、`tol`、`shape`、`flops`、`nbytes`。构造时校验名字是合法标识符、容差非负。 |
| [harness/registry.py](harness/registry.py) | 35 | 注册表。`register()` 防重复；`resolve(name)` 同时接受 pybind 名和 legacy 测试函数名（回退匹配 `spec.test_fn`）；首次访问时惰性 import `kernels` 完成注册。 |
| [harness/kernels.py](harness/kernels.py) | 53 | **唯一写形状的地方**。目前只有 `fused_ln_linear_gelu`：4096³ fp32、`eps=1e-5`、torch 参考（LayerNorm→Linear→GELU）、`tol=(1e-3,1e-3)`、`flops=2MNK`、`nbytes=3·MNK·4`。新增 kernel 就是往这加一个 `Spec`。 |
| [harness/measure.py](harness/measure.py) | 102 | 测量内核。`_event_time()` 用 CUDA Event 卡住单次调用（含 kernel 内部输出分配）；`measure()` 先 warmup 5 次，再按目标 200 ms 反推迭代数（夹在 20–200）；`summarize()` 出 n/median/trimmed_mean/min/max/P10/P90/MAD/CV；`aggregate()` 跨进程取 median-of-medians，并算 `spread_pct`。 |
| [harness/judge.py](harness/judge.py) | 69 | 判定纯函数，无 GPU 也能测。`noise_se_pct()`：多进程用 `spread/√procs`，单进程退化为 `cv/√n`；`decide()` 判据见 §5。 |
| [harness/store.py](harness/store.py) | 117 | 记录层。`fingerprint()` 抓 git commit/dirty、GPU 名/SM 数/最高 SM 时钟/driver、torch/cuda 版本、**`.so` 的 sha256 前 16 位**；`build_record()`/`append_record()`/`load_records()`/`last_record()`。 |
| [harness/report.py](harness/report.py) | 59 | 渲染层。把记录渲染成 Markdown 表（`vs 上一版` 自动算差值，dirty 提交加 `*`），只替换 `BENCH:TABLE:START/END` 之间的内容，**绝不覆盖手写散文**。 |
| [harness/cli.py](harness/cli.py) | 333 | 命令行与编排。6 个子命令；`run --procs N` 会真起 N 个**独立 OS 进程**（父进程再聚合，避免同进程暖机导致假稳定）；`--ref torch` 用同一套协议测参考实现；`--json -` 供父子进程通信。 |

### 3.3 `tools/`

| 文件 | 行数 | 作用 |
|---|---|---|
| [tools/gate.sh](tools/gate.sh) | 194 | 门禁执行器。一条命令串行跑完 B0–B5，输出逐门 PASS/FAIL；**只在全部门通过时**才追加 `BENCH.json` 并刷新 `BENCH.md`；从不 commit/push。参数：`--processes N`（默认 3）、`--no-build`、`--baseline`、`--min-gain PCT`（默认 8）。 |

### 3.4 `src/` —— kernel 本体

| 路径 | 作用 |
|---|---|
| `src/bindings.cu` | **注册中心**。顶部集中声明全部 `torch::Tensor xxx(...)` 签名；底部 `PYBIND11_MODULE` 里 92 个 `m.def`。新增 kernel 要改这里的**两处**：声明 + `m.def`（只加 `m.def` 会编译失败）。 |
| `src/cuda_kernels.cuh`（230 行） | 公共 device 工具（如 `div_up` 等）。**改它等于改 97 个 kernel**，必须补跑受影响 kernel 的正确性门。 |
| `src/utils.cuh`（168 行） | 另一组公共工具/宏，同样有全局影响。 |
| 其余 96 个 `*.cu` | 一题一文件（`gelu.cu`、`matmul`、`flash_attn`…）。统一模式：文件内 `static __global__` kernel + `xxx_cuda(...)` 启动封装 + 一个 `torch::Tensor xxx(...)` 的 pybind 包装（内部 `torch::empty` 分配输出）。**这些是唯一不许乱动的部分**，重构只围绕它们外围。 |

> 当前统计：97 个 `.cu`（96 kernel + bindings）、92 个 `m.def`、93 个 `test.py` 函数，
> 其中只有 **78 个三方同名**；27 个存在命名漂移（`vector_add`↔`vector_addition`、`matmul`↔
> `matrix_multiplication`、`fused_ln_linear_gelu`↔`fused_layernorm_linear_gelu`、`prefxix_sum` 拼写错误…）。
> 约定是**不改名**，用 `Spec.test_fn` 做映射。

### 3.5 `profile/` —— legacy

4 个 `profile_*.py`（matmul / lora_linear / attention_with_linear_bias / linear_recurrence）内联造输入 + warmup，
供 `ncu` 抓单 kernel。**只覆盖 92 个 kernel 里的 4 个**，所以新流程改用
`ncu ... python3 -m harness run <k> --ncu`（见 §6）。6 个 `*.ncu-rep` 是历史报告，已 gitignore。

### 3.6 生成物（都不进提交）

`build/`（setuptools 中间对象与 ninja 文件）、根目录 `leetgpu*.so`（`--inplace` 产物）、`__pycache__/`。

---

## 4. 构建是怎么工作的

### 4.1 `MAX_JOBS` 为什么不是核数

实测每个 nvcc 进程峰值 RSS **3.0–3.4 GiB**，且与 kernel 大小无关（大头是 `torch/extension.h`）：

| TU | 峰值 RSS | 单 TU 耗时 |
|---|---|---|
| `src/gelu.cu` | 3102 MiB | 61 s |
| `src/fused_ln_linear_gelu.cu` | 3103 MiB | 67 s |
| `src/bindings.cu` | 3430 MiB | 67 s |

所以本机（12 核 / 15.4 GiB 总内存）真正的瓶颈是内存：

```
jobs = clamp(1, (MemAvailable - 2 GiB) / 3.45 GiB, nproc)     # → 通常 2
```

- 显式 `MAX_JOBS=N` 优先；`LEETGPU_MEM_PER_JOB_MB` / `LEETGPU_MEM_RESERVE_MB` 可调。
- 全量重建 ≈ 97×65 s / 2 ≈ **50 min**，所以日常几乎都是增量构建。
- 构建后端是 torch 默认的 ninja（`ninja 1.13` 已装），`MAX_JOBS` 直接限制它的并行度。

---

## 5. 测量与判定规则

### 5.1 一次 `run` 的输出怎么读

以 `python3 -m harness run fused_ln_linear_gelu --procs 3 --ref torch` 为例（数字是真实输出）：

```
# proc 1/3: median=19.507ms min=19.055ms cv=3.77% n=20
# proc 2/3: median=19.344ms min=19.025ms cv=2.85% n=20
# proc 3/3: median=19.567ms min=19.218ms cv=3.87% n=20
# ref(torch): median=20.323ms min=20.093ms speedup=1.04x
# agg: procs=3 n=20 median=19.507ms min=19.025ms p90=21.086ms spread=1.1% worst_cv=3.87%
fused_ln_linear_gelu,4096x4096x4096,fp32,19.507,19.025
```

**逐字段**：

| 字段 | 含义 |
|---|---|
| `proc 1/3` | 第 1 个**独立 OS 进程**（共 3 个）。用独立进程而不是同一进程循环，是因为同进程会共享已暖机的上下文/显存池，结果会偏乐观。 |
| `median=19.507ms` | 该进程内 N 次计时的**中位数**。N 是自适应的：先试跑 1 次，再按「目标 200 ms ÷ 单次耗时」算，夹在 20–200 之间（这里是 20）。 |
| `min=19.055ms` | 该进程内**最快的一次**，最接近"无干扰"的下限。 |
| `cv=3.77%` | 该进程内逐次计时的**抖动**：`标准差 ÷ 均值`。注意它只描述*这一次运行内部*的稳定性。 |
| `n=20` | 该进程实际的计时次数（warmup 5 次不计入）。 |
| `ref(torch)` | 用**完全相同的协议**测 torch 参考实现（LayerNorm→Linear→GELU）。`speedup = 参考 median ÷ 聚合 median`，这里是 20.323 / 19.507 ≈ 1.04。 |
| `agg` | **跨进程聚合，这才是上报值**（`BENCH.json` 里存的就是它）。 |
| `procs=3` | 参与聚合的进程数。 |
| `n=20` | 每进程迭代数（各进程一致时取该值）。 |
| `median=19.507ms` | **median-of-medians**：把 3 个进程的 median 排序取中间那个（19.344 / **19.507** / 19.567）。中位数对偶发尖峰稳健，所以不取平均。 |
| `min=19.025ms` | 三个进程里最快的一次。 |
| `p90=21.086ms` | 三个进程里**最差的 p90**（尾部风险）。判定时用它防"中位数变好、尾部变差"。 |
| `spread=1.1%` | **跨进程离散度**：`(最大 median − 最小 median) ÷ 聚合 median`。它衡量"换一次运行会差多少"，判定用的噪声 SE 就由它推出来。 |
| `worst_cv=3.87%` | 三个进程里最大的 cv，仅供诊断——**它不参与判定**。 |
| 最后一行 `name,shape,dtype,median_ms,min_ms` | 机器可读的 CSV，给脚本/管道用；数值与 `agg` 的 median/min 一致。 |

> 最容易混的一对：**`cv` 是"这一次跑得稳不稳"，`spread` 是"下次再跑会不会变"**。
> 做优化对比时你真正关心的是 `spread`（因为对比的是两次不同的运行），所以判定阈值用 `spread/√procs`，
> 而不是看起来更大的 `cv`——用 `cv` 会把门槛错误抬高（实测 cv≈3.4% 会让门槛从 8% 涨到 10.3%）。

### 5.2 判定公式

`harness compare`（以及 `gate.sh` 的 B3）用：

```
gain      = 1 - cand.median / base.median
noise_SE  = spread/√procs        （多进程）  或  cv/√n   （单进程）
threshold = max(8%, 3×noise_SE)
ACCEPT  ⇔  gain ≥ threshold  且  cand.p90 ≤ base.p90 × 1.02
```

### 5.3 为什么门槛是 8%

同版本 6 次独立进程的 median 在 20.70–22.75 ms（跨度 9.9%），2 进程 + 5% 门槛会误判；
3 进程后跨进程 spread 只有 0.2–1.1%，8% 已是约 4σ。`p90` 那一项防"中位数变好但尾部变差"。

**跨 session 无效**：同一份未改代码隔天量到 21.0 → 19.6 ms（≈7% 漂移，GPU 时钟/温度/宿主负载）。
判定前必须在同一 session 内重新冻结基线。

### 5.4 为什么 `test.py` 说快十几倍，`harness` 说只有 1.05 倍？

因为 `test.py` 量的是**冷启动 + 单次**，`harness` 量的是**稳态**。同一份代码、同一份输入，实测拆解：

| 参考实现里的操作 | 首次调用（冷） | 稳态 |
|---|---|---|
| `mean` | 26.17 ms | ~1 ms |
| `var` | 0.81 ms | ~1 ms |
| `x - mean` | 16.28 ms | — |
| `* gamma` | 14.21 ms | — |
| `/ sqrt(var+eps)` | 57.57 ms | — |
| `+ beta` | 3.32 ms | — |
| `x_norm @ W` | **152.79 ms** | 16.7 ms |
| `gelu(+bias)` | 14.35 ms | — |
| **整段合计** | **≈285 ms**（整段实测 311.67 ms） | **20.38 ms** |

- 冷的那 285 ms 是**进程级的一次性成本**：cuBLAS handle/算法选择（matmul 152 ms）、每个新 kernel 首次
  launch 的动态模块加载（`/sqrt` 57 ms、elementwise 各十几 ms）。它们只付一次，与 kernel 写得好不好无关。
- kernel 自己也会冷：单次冷 24.24 ms vs 稳态 19.48 ms（约 1.24×）；而 torch 参考是 311.67 → 20.38（约 15×）。
  于是**冷 vs 冷的比值被放大成 12.9×**，这就是"快很多"的来源。
- 稳态对比（CUDA Event + warmup 5 + 20 次中位数）：kernel 19.48 ms vs torch 20.38 ms → **1.05×**。

结论：冷启动数字可以解释"为什么第一次跑很慢"，但不能用来比较实现优劣；任何性能结论都必须走
`harness run`（warmup + 多次取中位数），否则"优化"可能只是把冷启动挪走了。

> 附注：`test.py` 的 `@timeit` 用 `time.time()` 且不 `torch.cuda.synchronize()`，异步调用会让数字偏**小**
> 而不是偏大，所以 334 ms 的成因是冷启动，不是异步假象。

---

## 6. 常见任务

### 6.1 新增一个 kernel

1. 定名 `<name>`（与 `src/<name>.cu`、`m.def("<name>")`、`Spec.name` 三者一致）。
2. `harness/kernels.py` 加 `Spec`：`make_inputs`、`reference`、`tol`、`shape`/`flops`/`nbytes`。
3. 写 `src/<name>.cu`；`src/bindings.cu` 加声明 + `m.def`。
4. `bash tools/gate.sh <name> --baseline` → 全 PASS 才算基线成立。
5. `git commit -m "feat(<name>): naive kernel (median X ms)"`（`BENCH.json`/`BENCH.md` 已由 gate 写好）。

### 6.2 优化一个 kernel

```bash
# 0) 工作区干净，HEAD 即基线
git status --porcelain
# 1) 冻结基线（同 session）
python3 -m harness run <name> --procs 3
# 2) 剖析：读 SOL、occupancy、访存
sudo ncu --set full -s 5 -c 1 -o /tmp/report_<name> python3 -m harness run <name> --ncu
# 3) 先写 3 条假设（改动→预期指标→预估收益），等确认
# 4) 一次只改一条
bash tools/gate.sh <name>
# 5) 过门才提交；不过则回退（可能不止一个文件）
git restore --source=HEAD --worktree -- src/<name>.cu src/bindings.cu src/cuda_kernels.cuh src/utils.cuh
```

### 6.3 只测不记账

```bash
python3 -m harness run <name>                 # CSV: name,shape,dtype,median_ms,min_ms
python3 -m harness run <name> --procs 3 --ref torch
python3 -m harness run <name> --markdown      # 打印一行 BENCH.md 表格
python3 -m harness selftest                   # 判定逻辑自测，不需要 GPU
python3 -m harness report                     # 用 BENCH.json 重刷 BENCH.md 表格
```

### 6.4 排查

| 现象 | 原因 / 处理 |
|---|---|
| `CUDA error 304` | 沙箱限制，改 Full access；不要改代码。 |
| 构建 OOM | 别手动调大 `MAX_JOBS`；用 `MAX_JOBS=1` 兜底。 |
| `unknown kernel` | `harness/kernels.py` 里没有对应 `Spec`。 |
| 测出来"变快了"却没过门 | 阈值含 3×噪声项，且跨 session 不可比；同 session 重冻基线。 |
| `BENCH.md` 行带 `*` | 记录时工作区脏，commit 不精确对应被测代码；提交后重跑。 |

---

### 6.5 哪些 kernel 已经能走 harness？

**目前只有 1 个：`fused_ln_linear_gelu`。** 其余 91 个 kernel（`m.def` 共 92 个）还不可以：

| 命令 | 结果 |
|---|---|
| `python3 -m harness run gelu` | `unknown kernel: gelu` + 可用列表，退出码 **2** |
| `bash tools/gate.sh gelu` | `FAIL B0 registry: 'gelu' has no Spec`，退出码 **1**（连构建都不会跑） |
| `python3 -c 'import test; test.gelu()'` | **仍然可用**（legacy 路径），但只有打印、无判定、无记账 |

为什么不自动全接：每个 kernel 的输入形状、dtype、eps/scale 参数、参考实现都不同（有 fp16、int8、带 mask 的
attention、带随机采样的 top-p 等），必须逐个写 `Spec`（约 15–25 行）。自动生成很容易造出"形状对了但语义错了"
的假基线，比没有基线更危险。

怎么接：按 §6.1，或在 `harness/kernels.py` 里照 `fused_ln_linear_gelu` 那段加一个 `register(Spec(...))`，
形状抄 `test.py` 里对应函数的输入构造即可（**只抄一次**，之后不再双写）。

---

## 7. 当前状态与待办

- 已注册 Spec：**1 个 / 92 个 kernel**（`fused_ln_linear_gelu`）；`BENCH.json` 有 2 条记录（`1d61067*`、`0c4ae54`）。
- `harness suite` 能跑，但当前只会跑这 1 个已注册的 Spec。
- 未实现：多形状 sweep（`Spec.variants` 已删除，避免空壳 API）。
- 下一步二选一：
  1. 按工作流 B 对 `fused_ln_linear_gelu` 做一轮真实优化（先出 3 条假设再改码）；
  2. 给 2–3 个待优化 kernel 补 `Spec`，把 `suite` 铺开。
