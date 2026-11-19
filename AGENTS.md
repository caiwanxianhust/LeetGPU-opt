# AGENTS.md

本文件给在此仓库工作的 agent / 人看的约定。命令与路径均基于仓库内实际文件（`setup.py`、`test.py`、`profile/*.py`、`.gitignore`）核对过，不要凭记忆改动。

## 项目是什么

单卡 CUDA kernel 练习/优化仓库（LeetGPU 题目集）。每个 kernel 一个 `src/*.cu` 文件，统一由 `src/bindings.cu` 的 `PYBIND11_MODULE` 通过 `m.def(...)` 暴露为 Python 扩展 `leetgpu`；`test.py` 用 PyTorch 参考实现做正确性对比并打印耗时；`profile/` 下用 Nsight Compute 对单 kernel 做性能剖析。

## 目录约定

```
README.md       # 教程：目录与文件逐个说明、数据流、常见任务
setup.py        # 唯一构建入口（内存感知 MAX_JOBS + CUDAExtension）
test.py         # 【legacy】存量 kernel 的正确性测试（非 pytest）
harness/        # 评估框架：spec / registry / measure / judge / store / report / cli
  kernels.py    #   ← 新 kernel 的输入规格、参考实现、容差、成本模型都写这里
bench.py        # 兼容壳：转发到 python3 -m harness run
BENCH.json      # 机器可读测量历史（含 so_sha256、commit、GPU 指纹）
BENCH.md        # 人读账本；表格由 harness report 生成，散文手写
tools/gate.sh   # 门禁：B1 build → B2 check → B3 run+compare → B4 卫生 → B5 记账
src/            # 97 个 *.cu（一个题目一个文件）+ cuda_kernels.cuh + utils.cuh
src/bindings.cu # 顶部集中声明全部签名 + PYBIND11_MODULE 里 92 个 m.def 注册点
profile/        # 【legacy】4 个 ncu 剖析脚本；新 kernel 用 harness run --ncu 代替
build/          # setuptools 中间产物（已 gitignore）
leetgpu.cpython-310-x86_64-linux-gnu.so  # 就地构建的扩展（已 gitignore）
```

没有 Makefile / CMakeLists.txt / pyproject.toml。**构建只能走 `setup.py`**。结构与每个文件的职责见 `README.md`。

新增一个 kernel 的完整改动面见「工作流 A」。`setup.py` 用 `glob.glob("src/*.cu")` 全量编译，所以 `src/` 里放进去就会参与编译；注意改 `cuda_kernels.cuh` / `utils.cuh` 会波及全部 97 个 kernel。

## 环境要求

| 项 | 值 |
|---|---|
| nvcc | CUDA 12.8（V12.8.93），`/usr/local/cuda-12.8` |
| GPU / 架构 | RTX 4060 Laptop 8GB，compute capability 8.9 → `-arch=sm_89` |
| Python | `python3` = 3.10（`python` 不在 PATH；扩展是 `cpython-310` ABI，必须用 python3.10） |
| PyTorch | 2.9.0+cu128 |
| CUTLASS | `/usr/local/cutlass/include`（`setup.py` 里硬编码了 include 路径） |
| 宿主 | WSL2（`5.15.167.4-microsoft-standard-WSL2`） |

设备与驱动侧正常（`nvidia-smi` 可用），但**构建 / 测试 / 跑分必须在 Full access 权限下进行**，原因见家规 4。

## 构建

```bash
cd /home/houlai/resumeProject/hpc/leetgpu
python3 setup.py build_ext --inplace
```

- `MAX_JOBS` 由可用内存自动算：`jobs = (MemAvailable - 2 GiB) / 3.45 GiB`，夹在 `[1, nproc]`。
  实测每个 nvcc 峰值 RSS **3.0–3.4 GiB**（大头是 `torch/extension.h`，与 kernel 大小无关；单个 TU 约 65 s），
  本机 12 核 / 10.3 GiB 可用 → 自动选 **2**。**并行度受内存限制，不是核数**，手动调到 4 以上会 OOM（历史故障）。
  显式 `MAX_JOBS=N` 优先；`LEETGPU_MEM_PER_JOB_MB` / `LEETGPU_MEM_RESERVE_MB` 可调。
- `setup.py` 里用 `glob.glob("src/*.cu")` 全量编译；全量重建 ≈ 97 × 65 s / 2 ≈ 50 min，增量构建是常态。
- 固定 `-arch=sm_89 -std=c++17`。换 GPU 架构需改 `setup.py`。
- `--inplace` 把 `leetgpu*.so` 放到仓库根，`test.py` 才能直接 `import leetgpu`。

## 测试

没有 pytest，也没有 `test_*.py` 收集机制。**首选按模块导入调用，不要改 `test.py`**：

```bash
cd /home/houlai/resumeProject/hpc/leetgpu
python3 -c 'import test; test.<name>()'
# 例：
python3 -c 'import test; test.fused_layernorm_linear_gelu()'
```

**已验证可用（勿再怀疑）**：`import test` 无副作用——AST 检查显示模块级只有 5 个 `import`、1 个 `ImportFrom`、93 个 `FunctionDef`、1 个 `ClassDef`，唯一的可执行块是第 4293 行的 `if __name__ == "__main__"`，导入时不会触发。实测 `python3 -c 'import test; test.fused_layernorm_linear_gelu()'` 正常跑通并打印 `All tests passed!`（本机 RTX 4060 Laptop，kernel 27.18 ms / PyTorch 334.67 ms）。因为 `python3 -c` 会把 cwd 放到 `sys.path[0]`，仓库根的 `leetgpu*.so` 可直接被 `test.py` 导入，无需 `PYTHONPATH`。

**仅当上述形式不可行时**（例如需要临时加打印、或要顺序跑多个函数），才改 `test.py`：把 `if __name__ == "__main__"`（第 4293 行起）里目标函数那行取消注释、其余保持注释，然后 `python3 test.py`。当前该块里只有第 4383 行的 `fused_layernorm_linear_gelu()` 是激活的。改这一块属于临时调试，提交前请还原，并在回复中说明改了哪一行。

判读方式（**门禁必须机器可判**）：
- 新增 kernel 一律写 `assert compare_tensors(...)`，失败即 AssertionError → 非零退出。
- 存量测试有两派：早期 28 处 `assert torch.allclose(...)`（全文件共 31 条 assert）可靠；后期 71 处裸
  `compare_tensors(...)` 调用丢弃返回值，**比较失败也 exit 0**，只能靠输出里的 `mismatch` / `Maximum difference`
  关键字识别。`tools/gate.sh` 的 B2 走 `harness check`，不要只看退出码。
- 测试默认要求 GPU 可用，故同样受家规 4 约束。

## 跑分 / 剖析

测量入口是 `harness`（`bench.py` 只是兼容壳）：

```bash
python3 -m harness run <name> --procs 3 --ref torch   # 3 个独立进程 + torch 参考
python3 -m harness run <name> > out.csv               # 单行 CSV: name,shape,dtype,median_ms,min_ms
python3 -m harness check <name>                       # 只做正确性（spec 的 reference + tol）
python3 -m harness compare base.json cand.json        # 判定 accept/reject（8% + 3×SE + p90 尾部）
python3 -m harness report                             # 用 BENCH.json 重写 BENCH.md 表格
python3 -m harness selftest                           # 判定逻辑单测，不需要 GPU
```

新 kernel 的**输入规格、torch 参考、容差、成本模型**全部写在 `harness/kernels.py` 的 `Spec` 里；
形状只写一处，不再和 `test.py` 双写。

剖析用 ncu 包 `harness run --ncu`（warmup 5 次后只跑 1 次；`-s 5 -c 1` 只采最后一次 launch）：

```bash
sudo ncu --set full -s 5 -c 1 -o /tmp/report_<name> python3 -m harness run <name> --ncu
# kernel 内部若 launch 多个 kernel（如 twopass 实现），把 -c 调大
```

`profile/*.py` 是 **legacy**（只覆盖 matmul / lora_linear / attention_with_linear_bias / linear_recurrence
这 4 个），新 kernel 不要再加剖析脚本。

`test.py` 里的耗时是单次 `time.time()` 墙钟，噪声大；**判定只认 `harness` + `tools/gate.sh`**。

## 家规

1. **产物必须被 `.gitignore` 覆盖且不进提交。** 首选 `build/`；允许两个例外：`build_ext --inplace` 产出的根目录 `leetgpu*.so`，以及 `profile/*.ncu-rep`。新增任何产物类型时，同步把对应路径补进 `.gitignore`。
2. **临时脚本与日志写 `/tmp`。** 不要在仓库里留 `tmp_*.py`、`*.log`、调试打印文件等一次性产物。
3. **提交前运行 `git status --porcelain` 确认没有多余的未跟踪文件。** 期望输出为空，或只剩本次有意提交的文件；出现意料外的未跟踪文件先清理再提交。
4. **构建 / 测试 / 跑分必须在 Full access 权限下进行。** workspace-write 沙箱下 `torch.cuda.is_available()` 返回 `False` 并报 `CUDA error 304: OS call failed or operation not supported on this OS`（沙箱挡住了 `/dev/dxg` 的设备 ioctl），而 `nvidia-smi` 一切正常。实测同一命令提到 Full access 后立刻变成 `cuda_avail = True`、设备名正常、kernel 跑通。看到 304 先提权重试，不要改代码、不要当成 GPU 或驱动故障。
5. **性能结论必须是同形状、同精度、同 GPU 的前后对比。** shape / dtype / 设备任一不同就不允许下"更快/更慢"的结论；要写明对比依据（ncu 指标，或多次计时的中位数），`test.py` 的单次墙钟只能用于粗筛。


## 工作流 A：新增 kernel

顺序不可颠倒；**没有测试就不写 kernel**。

1. 定名 `<name>`（snake_case）。该名要同时是：`src/<name>.cu` 文件名、`bindings.cu` 里的
   `m.def("<name>", ...)`、`harness/kernels.py` 里 `Spec.name`。三者必须完全一致。
   （存量有 27/92 不同名，如 `vector_add`↔`vector_addition`、`fused_ln_linear_gelu`↔
   `fused_layernorm_linear_gelu`；**不要改名**，用 `Spec.test_fn` 字段做映射。）
2. 先在 `harness/kernels.py` 写 `Spec`：`make_inputs(seed)` 固定 seed 造输入、
   `reference` 是 torch 参考、`tol` 是容差、`shape`/`flops`/`nbytes` 供报告。
   正确性由 `python3 -m harness check <name>` 机器判定（allclose）。
   若还想兼容 legacy `test.py`，再补一个同名测试函数并用 `assert compare_tensors(...)`——
   **不要为凑通过放宽 atol/rtol**。
3. 写 naive 实现 `src/<name>.cu`；然后在 `bindings.cu` 里改**两处**：
   ① 顶部声明区加一行 `torch::Tensor <name>(...);`（否则 `m.def` 引用不到，B1 直接编译失败）；
   ② `PYBIND11_MODULE` 里加一行 `m.def("<name>", &<name>, "...");`。
4. 跑门禁：`bash tools/gate.sh <name> --baseline`
   （B1 构建 + B2 `harness check` + B3 测量 + B4 卫生 + B5 记账，一条命令给出 PASS/FAIL）。
5. **门 A1（正确性）**：gate 的 B2 必须 PASS（`harness check` 退出 0，`max|diff|` 在 `tol` 内）。
6. **门 A2（基线）**：gate 的 B3 以 `--baseline` 模式 PASS，记录 median-of-medians / min。
7. 提交：`git commit -m "feat(<name>): naive kernel (median X ms)"`。
   gate 通过后已自动写入 `BENCH.json` 并刷新 `BENCH.md` 表格，随同提交即可。
   naive 版本必须先入库——它是后面每次优化的回退点。
8. 进入工作流 B。

## 工作流 B：优化 kernel（单变量、可回退）

0. 前置：工作区干净；当前 HEAD 就是基线（`git status --porcelain` 为空）。
1. 冻结基线：`python3 -m harness run <name> --procs 3`，与 `BENCH.json` 里该 kernel 的最后一条
   在噪声带内才开工（`python3 -m harness compare <那条> <本次>` 会直接告诉你是否可比）。
2. 剖析：`sudo ncu --set full -s 5 -c 1 -o /tmp/report_<name> python3 -m harness run <name> --ncu`
   读 SOL：受限于 Compute 还是 Memory？achieved occupancy？有没有 bank conflict / 未合并访存？
3. **先假设后改码**：写 3 条「改动 → 预期哪个指标变化 → 预估收益」，等用户确认。
4. 一次只实现一条假设（单变量）。
5. 每改完立刻过门：`bash tools/gate.sh <name>`
   - **B1 编译**：退出 0。
   - **B2 正确性**：`harness check` 退出 0。**禁止放宽 atol/rtol 来凑通过。**
   - **B3 性能**：3 进程 median-of-medians，由 `harness compare` 判定
     （`gain ≥ max(8%, 3×噪声SE)` 且 `p90` 不劣化 2%）。
   - **B4 卫生**：只有预期路径改动。
6. 通过则提交：`git commit -m "perf(<name>): <改动> (X → Y ms)"`，**一条假设一条 commit**。
   不通过则回退——改动可能不止一个文件，别只 checkout 一个：
   `git restore --source=HEAD --worktree -- src/<name>.cu src/bindings.cu src/cuda_kernels.cuh src/utils.cuh`
   然后在回复里写下"假设 Hn 无效"。
7. gate 通过时已自动追加 `BENCH.json` 并刷新 `BENCH.md` 表格；你只需把「噪声带」表里
   该 kernel 的实测区间补上（人写部分不会被 `harness report` 覆盖）。

> 硬约束：
> - 改 `cuda_kernels.cuh` / `utils.cuh` 视为影响全部 97 个 kernel，必须补跑受影响 kernel 的 B2。
> - B1 与 B2/B3 必须在**同一次 gate 运行**内串行完成。`.so` 比源码旧、或 `git checkout` 刷新 mtime，
>   都会让 mtime 判据失效——禁止拿隔夜的 `.so` 下结论。

## 门与记录

| 门 | 命令 | 通过标准 |
|---|---|---|
| B0 注册 | gate 内 `registry.resolve(<name>)` | kernel 有 Spec |
| B1 编译 | `python3 setup.py build_ext --inplace` | 退出 0 |
| B2 正确性 | `python3 -m harness check <name>` | 退出 0（allclose 在 `tol` 内） |
| B3 性能 | `harness run --procs 3` + `harness compare` | `gain ≥ max(8%, 3×SE)` 且 p90 不劣化 2% |
| B4 卫生 | `git status --porcelain` | 只有预期路径 |
| B5 记录 | 自动追加 `BENCH.json` + 刷新 `BENCH.md` | 仅在全部门 PASS 时写 |

- 一条命令跑完全部门：`bash tools/gate.sh <name>`（新 kernel 加 `--baseline`，只重测可加 `--no-build`）。
- 每个任务开始时，先读 `BENCH.json` 里该 kernel 的最后一条 + `git log --oneline -5`，不要靠对话记忆。
- 一次只动一个 kernel。顺手"优化"别的文件属于越界，先报告再等确认。
- 跑分数字必须同形状 / 同精度 / 同 GPU；单次墙钟只能粗筛。
- 测量路径不能改：`Spec.make_inputs` 造输入后**直接调 pybind 入口**，与真实用法一致（含 kernel 内部
  的输出分配）；把分配挪到调用方就不算优化。