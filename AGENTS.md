# AGENTS.md

本文件给在此仓库工作的 agent / 人看的约定。命令与路径均基于仓库内实际文件（`setup.py`、`test.py`、`profile/*.py`、`.gitignore`）核对过，不要凭记忆改动。

## 项目是什么

单卡 CUDA kernel 练习/优化仓库（LeetGPU 题目集）。每个 kernel 一个 `src/*.cu` 文件，统一由 `src/bindings.cu` 的 `PYBIND11_MODULE` 通过 `m.def(...)` 暴露为 Python 扩展 `leetgpu`；`test.py` 用 PyTorch 参考实现做正确性对比并打印耗时；`profile/` 下用 Nsight Compute 对单 kernel 做性能剖析。

## 目录约定

```
setup.py        # 唯一构建入口（setuptools + torch.utils.cpp_extension.CUDAExtension）
test.py         # 自测脚本（非 pytest）。~90 个 kernel 各一个测试函数
src/            # 97 个 *.cu（一个题目一个文件）+ cuda_kernels.cuh + utils.cuh
src/bindings.cu # PYBIND11_MODULE + 92 个 m.def 注册点
profile/        # ncu 剖析脚本 + *.ncu-rep 报告（已 gitignore）
build/          # setuptools 中间产物（已 gitignore）
leetgpu.cpython-310-x86_64-linux-gnu.so  # 就地构建的扩展（已 gitignore）
```

没有 README / Makefile / CMakeLists.txt / pyproject.toml。**构建只能走 `setup.py`**。

新增一个 kernel 的完整改动面：`src/<name>.cu` 实现 → `src/bindings.cu` 加一行 `m.def` → `test.py` 加同名测试函数（用 `python3 -c 'import test; test.<name>()'` 调用，不必改 `__main__`）。`setup.py` 用 `glob.glob("src/*.cu")` 全量编译，所以 `src/` 里放进去就会参与编译。

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

- `setup.py` 内部强制 `MAX_JOBS=1`（单进程编译，全量 97 个 `.cu` 会比较慢）。
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

判读方式：`compare_tensors()` 打 `All tests passed!` 即通过，否则打印 shape mismatch 或 mismatch 索引、最大误差。测试默认要求 GPU 可用，故同样受家规 4 约束。

## 跑分 / 剖析

```bash
ncu --set full -o profile/report_matmul python3 profile/profile_matmul.py
sudo ncu --set full -o profile/report_lora_linear python3 profile/profile_lora_linear.py
sudo ncu --set full -o profile/report_attention_with_linear_bias python3 profile/profile_attention_with_linear_bias.py
sudo ncu --set full -o profile/report_linear_recurrence --launch-timeout 60000 python3 profile/profile_linear_recurrence.py
```

`profile/*.py` 内含 `warmup()` + `profile()`，并自行把仓库根加入 `sys.path`，故可从任意目录运行。已知过时注释：`profile/profile_linear_recurrence.py` 里写的是 `python3 profile_kernels.py`（该文件不存在），应以上面最后一行为准。

`test.py` 里的耗时是单次 `time.time()` 墙钟（未做多轮取均值），噪声大，只适合粗看；严肃性能对比用 ncu。

## 家规

1. **产物必须被 `.gitignore` 覆盖且不进提交。** 首选 `build/`；允许两个例外：`build_ext --inplace` 产出的根目录 `leetgpu*.so`，以及 `profile/*.ncu-rep`。新增任何产物类型时，同步把对应路径补进 `.gitignore`。
2. **临时脚本与日志写 `/tmp`。** 不要在仓库里留 `tmp_*.py`、`*.log`、调试打印文件等一次性产物。
3. **提交前运行 `git status --porcelain` 确认没有多余的未跟踪文件。** 期望输出为空，或只剩本次有意提交的文件；出现意料外的未跟踪文件先清理再提交。
4. **构建 / 测试 / 跑分必须在 Full access 权限下进行。** workspace-write 沙箱下 `torch.cuda.is_available()` 返回 `False` 并报 `CUDA error 304: OS call failed or operation not supported on this OS`（沙箱挡住了 `/dev/dxg` 的设备 ioctl），而 `nvidia-smi` 一切正常。实测同一命令提到 Full access 后立刻变成 `cuda_avail = True`、设备名正常、kernel 跑通。看到 304 先提权重试，不要改代码、不要当成 GPU 或驱动故障。
5. **性能结论必须是同形状、同精度、同 GPU 的前后对比。** shape / dtype / 设备任一不同就不允许下"更快/更慢"的结论；要写明对比依据（ncu 指标，或多次计时的中位数），`test.py` 的单次墙钟只能用于粗筛。


## 工作流 A：新增 kernel

顺序不可颠倒；**没有测试就不写 kernel**。

1. 定名 `<name>`（snake_case）。`test.py` 的函数名、`src/<name>.cu`、`bindings.cu` 里的
   `m.def("<name>", ...)`、`bench.py` 里的 builder 名**必须完全一致**。
2. 先在 `test.py` 写测试函数：固定 seed 造输入 → 调 `leetgpu.<name>` → `compare_tensors()`
   对比同 dtype / 同 device 的 PyTorch 参考 → 打印耗时。
3. 写 naive 实现 `src/<name>.cu`；在 `bindings.cu` 的 `PYBIND11_MODULE` 里加一行 `m.def`；
   在 `bench.py` 里加对应 builder（形状与 `test.py` 一致）。
4. `python3 setup.py build_ext --inplace`
5. **门 A1（正确性）**：`python3 -c 'import test; test.<name>()'` → 必须 `All tests passed!`
6. **门 A2（基线）**：`python3 bench.py <name>` 跑 2 个独立进程，记录 median / min。
7. 提交：`git commit -m "feat(<name>): naive kernel (median X ms)"`。
   naive 版本必须先入库——它是后面每次优化的回退点。
8. 进入工作流 B。

## 工作流 B：优化 kernel（单变量、可回退）

0. 前置：工作区干净；当前 HEAD 就是基线。
1. 冻结基线：`python3 bench.py <name>` ×2 进程，median 与 `BENCH.md` 最后一行对得上才开工。
2. 剖析：
   `sudo ncu --set full -o /tmp/report_<name> python3 profile/profile_<name>.py`
   读 SOL：受限于 Compute 还是 Memory？achieved occupancy？有没有 bank conflict / 未合并访存？
3. **先假设后改码**：写 3 条「改动 → 预期哪个指标变化 → 预估收益」，等用户确认。
4. 一次只实现一条假设（单变量）。
5. 每改完立刻过门：
   - **门 B1**：编译通过。
   - **门 B2**：`python3 -c 'import test; test.<name>()'` 全过。**禁止放宽 atol/rtol 来凑通过。**
   - **门 B3**：`python3 bench.py <name>` ×2 进程，median 下降 ≥5% 且两个进程方向一致。
6. 通过则提交：`git commit -m "perf(<name>): <改动> (X → Y ms)"`，**一条假设一条 commit**。
   不通过则 `git checkout -- src/<name>.cu` 回退，在回复里写下"假设 Hn 无效"。
7. 提交后在 `BENCH.md` 追加一行。

## 门与记录

| 门 | 命令 | 通过标准 |
|---|---|---|
| B1 编译 | `python3 setup.py build_ext --inplace` | 退出 0 |
| B2 正确性 | `python3 -c 'import test; test.<name>()'` | `All tests passed!` |
| B3 性能 | `python3 bench.py <name>` ×2 | median 降 ≥5%，两进程一致 |
| B4 卫生 | `git status --porcelain` | 只有本次预期的改动 |
| B5 记录 | `BENCH.md` 末尾追加一行 | 一次提交 = 一行 |

- 每个任务开始时，先读 `BENCH.md` 最后 5 行 + `git log --oneline -5`，不要靠对话记忆。
- 一次只动一个 kernel。顺手"优化"别的文件属于越界，先报告再等确认。
- 跑分数字必须同形状 / 同精度 / 同 GPU；单次墙钟只能粗筛。
- 测量路径不能改：`bench.py` 走的就是 `test.py` 的调用路径（含输出分配），
  把分配挪到调用方就不算优化。