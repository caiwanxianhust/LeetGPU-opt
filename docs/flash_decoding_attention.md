# flash_decoding_attention：从 14.88 ms 到 10.37 ms

本文记录 `src/flash_decoding_attention.cu` 的每一次结构性改动：**问题是什么、代码改前改后长什么样、为什么这样能快、量到了什么**。
代码段是取自对应 commit（`git show <commit>:src/flash_decoding_attention.cu`）的节选，语句逐字对应；
中文注释是讲解补充，为篇幅省略的部分用 `/* ... */` 标出。

> **数字来源与可比性**：`本机` 指 RTX 4060 Laptop（8 GB、24 SM、272 GB/s、sm_89），
> `T4` 指 LeetGPU 评测机（Tesla T4：16 GB、40 SM、GDDR6 320 GB/s、L2 4 MB、无 `cp.async`）。
> 两个平台的数字**不可互相换算**：同一份代码在本机 97% DRAM 打满时测不出差异，在 T4 上却能差 30%。
> T4 的数字是平台上测得的 Runtime，本机的数字是 `python3 -m harness run --procs 3` 的 median-of-medians。

---

## 1. 问题与契约

```
Q [T, H, D]   K [S, H, D]   V [S, H, D]   →   O [T, H, D]
O[t,h,:] = softmax_s(Q[t,h,:] · K[s,h,:] / sqrt(D)) · V[s,h,:]
```

- 约束：`1 ≤ T ≤ 16`、`1 ≤ S ≤ 131072`、`1 ≤ H ≤ 64`、`1 ≤ D ≤ 256`，输入 ∈ [-6, 6]。
- 性能测量形状：`T = 1, S = 65536, H = 32, D = 128`，fp32。
- 该形状的**必读字节数**：`K + V = 2 × S × H × D × 4 B = 2.147 GB`。
- 正确性判据：`harness check`，`torch.allclose(atol=1e-3, rtol=1e-3)`，实测 `max|diff| = 3.68357e-05`。

因为 `T` 很小、`S` 很大，这是典型的 **decode 阶段 attention**：一个 query 要扫 6.5 万行 K/V，
算力几乎为零，时间全在把 2.147 GB 搬进来。所以后面所有的优化都在两个方向上：
**把访存做宽/做连续**、**让更多 load 同时在飞**。

---

## 2. 朴素实现：两趟 flash decoding（`36b88aa`，T4 14.8822 ms）

### 2.1 为什么需要两趟

`S = 65536`，一个 block 不可能把整段 K/V 读进来再算 softmax，所以切块：

1. **第一趟**：把 S 切成 `num_chunks = ceil(S / 128)` 个 chunk，每个 block 处理一个 (chunk, head, query)，
   对 chunk 内的行做完整 softmax，输出**归一化后的 partial 结果** `O_chunk[T, H, num_chunks, D]` 和
   **log-sum-exp** `lse[T, H, num_chunks]`（`lse = max + log(sum(exp))`）。
2. **第二趟**：每个 (head, query) 一个 block，用 `lse` 把 `num_chunks` 个 partial 重新加权合并。

`lse` 是这整套结构的关键：有了它，两个 chunk 的 partial 可以在不知道彼此存在的情况下合并，
因为 `exp(lse_i - global_lse)` 正好等于「chunk i 在全序列 softmax 里占的权重」。

### 2.2 第一趟：chunk kernel

```cpp
const int chunk_idx = blockIdx.x;      // 128 行一段
const int head_idx  = blockIdx.y;
const int query_idx = blockIdx.z;

extern __shared__ float s_buf[];
float* s_Q = s_buf;
for (int i = threadIdx.x; i < D; i += blockDim.x) s_Q[i] = Q_ptr[i];
__syncthreads();

__shared__ float s_qk[CHUNK_SIZE];

// QK：一个 warp 负责一行，lane 沿 D 方向跨步
for (int row_id = warp_id; row_id < eff_chunk; row_id += (blockDim.x >> 5)) {
    float qk = 0.0f;
    for (int i = lane_id; i < D; i += 32) {
        qk += s_Q[i] * K_ptr[row_id * H * D + i];
    }
    qk = warp_all_reduce_sum(qk);
    if (lane_id == 0) {
        s_qk[row_id] = qk * softmax_scale;
    }
    __syncwarp();
}
__syncthreads();

// softmax：max 与 sum 都是 block 级归约
for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
    max_score = max(max_score, s_qk[i]);
}
max_score = block_all_reduce_max(max_score);

for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
    float exp_score = expf(s_qk[i] - max_score);
    s_qk[i] = exp_score;
    sum_exp_score += exp_score;
}
sum_exp_score = block_all_reduce_sum(sum_exp_score);

// PV：每个线程负责一个输出列，串行扫 128 行
for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float acc = 0.0f;
    for (int row_id = 0; row_id < eff_chunk; ++row_id) {
        float weight = s_qk[row_id] / sum_exp_score;
        float v = V_ptr[row_id * H * D + i];
        acc += weight * v;
    }
    output_ptr[i] = acc;
}

if (threadIdx.x == 0) lse[lse_offset] = max_score + logf(sum_exp_score);
```

### 2.3 第二趟：reduce kernel

```cpp
// Phase 0: lse 整块搬进 shared（只从 global 读这一次）
for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
    float lse_val = lse_ptr[i];
    max_lse = max(max_lse, lse_val);
    s_lse[i] = lse_val;
}
__syncthreads();

max_lse = block_all_reduce_max(max_lse);

float deno = 0.0f;
for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
    deno += expf(s_lse[i] - max_lse);
}
deno = block_all_reduce_sum(deno);
const float global_lse = max_lse + logf(deno);

// lse 原地改写成权重
for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
    s_lse[i] = expf(s_lse[i] - global_lse);
}
__syncthreads();

for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float acc = 0.0f;
    for (int row_id = 0; row_id < num_chunks; row_id++) {
        float w = s_lse[row_id];
        acc += O_chunk_ptr[row_id * D + i] * w;
    }
    O_ptr[i] = acc;
}
```

### 2.4 它的三个问题

| 问题 | 后果 |
|---|---|
| 每 lane 一次只取 **4 B**（`K_ptr[row_id * H * D + i]`，i 按 32 跨步） | 一行 512 B 由 32 lane 分摊成 32 个 4 B 请求，虽然合并但对每个 lane 只有 1 个在途 load |
| PV 是 `for (row_id = 0..127)` 的**串行累加**，`acc += weight * v` | 每 lane 在整段 PV 里只有 1 个 load 在飞，延迟完全暴露 |
| 分数走 `s_qk` shared + block 级 softmax + block 级归约 | 至少 4 次 `__syncthreads`，每行一次 `__shfl` 归约 |

在 4060 上它已经能到 `10.566 ms`（DRAM 96.85%），因为 4060 的 24 SM 对 272 GB/s 尚能喂饱；
**T4 是 40 SM 对 320 GB/s，单 SM 带宽更高、延迟也更长，于是瓶颈从带宽变成「在途 load 数量」**——
这正是 14.88 ms 的来源（只有 144 GB/s = 45% 峰值）。

---

## 3. 优化 1：D=128 快速路径（`8d9422c`，T4 **14.8822 → 10.56 ms**）

### 3.1 思路

`D = 128` 时一行正好是 **128 × 4 B = 512 B**，而 32 lane × `float4`(16 B) = 512 B：
**一个 warp 的一条 `float4` 读取就能铺满整行**。这带来三件事：

1. 访存从「32 个 4 B/lane」变成「1 个 16 B/lane」，一条指令覆盖 512 B；
2. 每 lane 天然拥有 4 个输出列，`D` 维的串行循环消失；
3. Q 只有 128 个 float，可以整体放进 4 个寄存器，不再每行读 shared。

行跨度 `H * D` floats = `32 × 128 × 4 B = 16 KB`，是 512 B 的整数倍，所以 `float4` 永远对齐——
这就是选 `D_SPECIALIZED = 128` 而不是任意 D 特化的原因。

### 3.2 改前（通用路径，见上一节）

```cpp
for (int i = lane_id; i < D; i += 32) qk += s_Q[i] * K_ptr[row_id * H * D + i];
```

### 3.3 改后（d128 路径）

```cpp
const int col = lane_id * COLS_PER_LANE;      // 4

// Q 行常驻 4 个寄存器
const float4 q = *reinterpret_cast<const float4*>(Q_ptr + col);

// 4 行预取：4 个 16 B load 先全部发出，再算 dot
const int64_t row_stride = static_cast<int64_t>(H) * D;          // H*128
const int64_t quad_stride = static_cast<int64_t>(NUM_WARPS) * row_stride;
int row_id = warp_id;
for (; row_id + 3 * NUM_WARPS < eff_chunk; row_id += 4 * NUM_WARPS) {
    const int64_t r = static_cast<int64_t>(row_id) * row_stride + col;
    const float4 k0 = __ldg(reinterpret_cast<const float4*>(K_ptr + r));
    const float4 k1 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + quad_stride));
    const float4 k2 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + 2 * quad_stride));
    const float4 k3 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + 3 * quad_stride));
    float qk0 = q.x * k0.x + q.y * k0.y + q.z * k0.z + q.w * k0.w;
    /* ... qk1..qk3 同理 ... */
    qk0 = warp_all_reduce_sum(qk0);
    /* ... */
    if (lane_id == 0) {
        s_qk[row_id] = qk0 * softmax_scale;
        /* ... */
    }
}
```

PV 阶段同样是 4 行预取 + 每 lane 一个 `float4` 累加器：

```cpp
const float4 v0 = __ldg(reinterpret_cast<const float4*>(V_ptr + r));
const float4 v1 = __ldg(reinterpret_cast<const float4*>(V_ptr + r + row_stride));
/* ... v2, v3 ... */
const float w0 = s_qk[row_id + 0];  // softmax 权重已在前面统一乘过 1/sum
acc.x += w0 * v0.x; acc.x += w1 * v1.x; /* ... */
```

两个小改动也在这一步：

- **normalize 一次**：`const float inv_sum = 1.0f / sum_exp_score; s_qk[i] *= inv_sum;`
  原来每个 (row, col) 对做一次除法（PV 循环里 128 次），现在只有 1 次。
- **reduce 也走 d128**：权重表只建一次（`s_w[i] = expf(s_w[i] - global_lse)`），4 个 warp 分摊
  `num_chunks` 个 partial，`float4` 累加 + `float4` 写回。

### 3.4 量到的结果

| 指标 | 改前（通用） | 改后（d128） |
|---|---|---|
| Executed Instructions | 274,253,680 | **81,838,080**（-70%） |
| 本机 DRAM Throughput | 96.85% | 97.12% |
| 本机 Duration | 8.71 ms | 8.68 ms |
| **T4 Runtime** | **14.8822 ms** | **10.56 ms**（-29%） |

本机几乎不动（DRAM 已经打满，指令数不是瓶颈），T4 上直接兑现——这印证了 T4 是
**指令/延迟受限**，砍掉 3.5 倍指令 + 每 lane 4 个在途 load 就是全部收益。

> 注：8 行预取（`ROWS_IN_FLIGHT = 8`）试过，寄存器 40 → 56，理论 occupancy 100% → 75%，
> 本机逐位不变，判断为「拿 25% 占用率换不确定收益」后回退。

---

## 4. 优化 2：head-major 块序（`965ced7`，T4 **无变化**）

### 4.1 假设

一个 block 只读一行里属于自己 head 的 512 B，而整行宽 `H × D × 4 = 16 KB`。
如果同时驻留的 block 覆盖同一窗口的不同 head，整行 16 KB 会被连续读出来，DRAM page 只开一次。
于是在 kernel 里把 `blockIdx.x` 和 `blockIdx.y` 的角色对调：

```cpp
// 改前：dim3 grid(num_chunks, H, T)
const int chunk_idx = blockIdx.x;
const int head_idx  = blockIdx.y;

// 改后：dim3 grid(H, num_chunks, T)
const int chunk_idx = blockIdx.y;
const int head_idx  = blockIdx.x;
```

### 4.2 结果：假设被否定

T4 上仍是 10.56 ms，逐位相同。说明 T4 的 DRAM page 局部性**不是**瓶颈（否则该改动必然有效）。
本机 ncu 也只是 96.52% → 96.63%。改动保留，因为它中性且顺序更自然，但**没有性能收益**。

---

## 5. 优化 3：一个 warp 独占一个 head，四个 head 合一块（`cff9277`，T4 **10.56 → 10.3947 ms**）

### 5.1 问题

上一版一个 block 负责 1 个 head 的 128 行，4 个 warp 把这 128 行**拆开**（warp w 处理第 w, w+4, w+8…行），
所以：块内要 block 级 softmax、warp 间 partial 要经 shared 合并、共 6 次 `__syncthreads`；
而且一个 block 每行只读 512 B，整行其余 15.5 KB 由别的 block 去读。

### 5.2 改法

让 **4 个相邻 head 进同一个 block，每个 warp 独占其中一个 head 的整个 chunk**：

```cpp
constexpr int HEADS_PER_BLOCK = 4;
const int warp_id = threadIdx.x >> 5;
const int head_idx = blockIdx.x * HEADS_PER_BLOCK + warp_id;   // warp 与 head 一一对应
const int col = lane_id * COLS_PER_LANE;

__shared__ float s_scores_all[HEADS_PER_BLOCK][CHUNK_SIZE];
float* s_scores = s_scores_all[warp_id];      // 每个 warp 自己的记分区，无竞争
```

调度变成：

```cpp
if (D == D_SPECIALIZED) {
    if (H % HEADS_PER_BLOCK == 0) {
        dim3 grid(H / HEADS_PER_BLOCK, num_chunks, T);   // 4 个 head 一块
        flash_decoding_chunked_attention_kernel_d128_heads4<<<grid, block, 0, stream>>>(...);
    } else {
        dim3 grid(H, num_chunks, T);                     // H 不是 4 的倍数，回退
        flash_decoding_chunked_attention_kernel_d128<<<grid, block, 0, stream>>>(...);
    }
}
```

关键收益有三条：

1. **行内连续 2 KB**：4 个相邻 head 的 K 切片在内存里相邻（`4 × D × 4 B`），一次读出来；
   H=32 时 8 个相邻 block 覆盖整行 16 KB。
2. **零 block 屏障**：一个 head 的 chunk 全在一个 warp 手里，QK → softmax → PV 全在 warp 内，
   只用 `__syncwarp()`，块级归约全部消失。
3. **块数 16384 → 4096**：每块 4 倍工作量，调度与波次开销下降。

### 5.3 结果

- 本机：11.80 / 11.83 / 11.84 ms（降频态，与上一版逐位相同——4060 早已打满）
- T4：**10.56 → 10.3947 ms**（-1.6%）
- ncu（本机）：Grid 4096、Registers 40、理论 occupancy 100%、实测 97.24%、DRAM 96.63%

---

## 6. 优化 4：online softmax，K/V 单趟（`0707662`，T4 **10.3947 → 10.3695 ms**）

### 6.1 问题

优化 3 仍然是两趟：先用 128 行 K 算完 128 个分数，再从 shared 读分数、扫 128 行 V。
于是**每个 lane 在任一时刻只有一个流**（先全是 K load，后全是 V load），
且 128 个分数要在 shared 里存一遍、读两遍。

### 6.2 改法：边读边归一化

把 V 行和 K 行放在同一个迭代里读，用 online softmax 维护 `(max, sum, acc)` 三元组：

```cpp
float row_max = -INFINITY;
float row_sum = 0.0f;
float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

int r = 0;
for (; r + 1 < eff_chunk; r += 2) {
    const int64_t base = static_cast<int64_t>(r) * row_stride + col;
    const float4 k0 = __ldg(reinterpret_cast<const float4*>(K_ptr + base));
    const float4 v0 = __ldg(reinterpret_cast<const float4*>(V_ptr + base));
    const float4 k1 = __ldg(reinterpret_cast<const float4*>(K_ptr + base + row_stride));
    const float4 v1 = __ldg(reinterpret_cast<const float4*>(V_ptr + base + row_stride));

    float s0 = q.x * k0.x + q.y * k0.y + q.z * k0.z + q.w * k0.w;
    s0 = warp_all_reduce_sum(s0) * softmax_scale;

    const float m0 = max(row_max, s0);
    const float a0 = expf(row_max - m0);       // 旧 accumulate 的缩放系数
    const float b0 = expf(s0 - m0);            // 新行的权重
    row_max = m0;
    row_sum = row_sum * a0 + b0;
    acc.x = acc.x * a0 + b0 * v0.x;
    acc.y = acc.y * a0 + b0 * v0.y;
    acc.z = acc.z * a0 + b0 * v0.z;
    acc.w = acc.w * a0 + b0 * v0.w;
    // 第 r+1 行用 a1/b1 同理
}

const float inv_sum = 1.0f / row_sum;
acc.x *= inv_sum;
acc.y *= inv_sum;
acc.z *= inv_sum;
acc.w *= inv_sum;
*reinterpret_cast<float4*>(output_ptr + col) = acc;

if (lane_id == 0) {
    lse[(query_idx * H + head_idx) * num_chunks + chunk_idx] = row_max + logf(row_sum);
}
```

数学上等价于 softmax：`acc` 始终是「以当前 `row_max` 为基准的归一化前累计」，
遇到更大的分数就整体乘 `exp(old_max - new_max)` 缩放，`row_sum` 同步缩放。
数值上因为每步都减了 `row_max`，指数项永远 ≤ 1，不会溢出。

### 6.3 结果

| 指标 | 优化 3（两趟） | 优化 4（单趟） |
|---|---|---|
| 每 lane 在途 load | 4 个 K（或 4 个 V） | 2 行 × (K + V) = 4 个，**两个流同时在飞** |
| 本机 DRAM | 96.63% | **97.47%** |
| Executed Instructions | 79.2M | 118M（多 49%，`expf` 变多） |
| Compute (SM) | — | 8.32% |
| 本机 Duration | — | 8.65 ms |
| **T4 Runtime** | **10.3947** | **10.3695 ms** |

指令变多但仍然更快，因为 Compute 只有 8.32%——它依旧是纯带宽/延迟受限，多出来的 `expf` 不花钱。
代价是寄存器 40 → 43，occupancy 从 97% 掉到 82.7%（本机 ncu），T4 上这 15 个点没有吃掉收益。

---

## 7. workspace 分配的三代演进（"为什么只申请不释放"）

两趟结构需要一块 **per-chunk partial + lse** 的临时缓冲，容量：

```
T · H · ceil(S / 128) · (D + 1)   floats
```

| 场景 | 形状 | 容量 |
|---|---|---|
| 性能测量形状 | T=1, S=65536, H=32, D=128 | **8.06 MiB** |
| 典型 decode | T=64, S=8192, H=32, D=128 | **64.5 MiB** |
| 契约上限 | T=16, S=131072, H=64, D=256 | **1.004 GiB** |

### 第一代：每次调用 `cudaMalloc` + `cudaFree`（`36b88aa`）

```cpp
cudaMalloc(&d_buf, buf_size);
flash_decoding_chunked_attention<<<...>>>(...);
cudaFree(d_buf);
```

实测代价：8 MiB 的 `cudaMalloc + cudaFree` 中位 **0.582 ms/次**（ctypes 直测 libcudart）；
`cudaFree` 还必须确认没有在飞 kernel 才敢解映射（nsys 里 `cudaFree` 中位 9.42 ms，主要在等设备）。
本机端到端 9.93 ms 里有约 **1.2 ms** 花在这上面，而且 malloc 的最坏延迟到过 7.4 ms（长尾噪声）。

### 第二代：PyTorch caching allocator（`cad05ba`）

```cpp
torch::Tensor scratch = torch::empty({scratch_elems}, Q.options());
float* d_O_chunk = scratch.data_ptr<float>();
float* d_lse = d_O_chunk + O_chunk_elems;
```

allocator 在进程内维护 free-list，同尺寸请求直接命中，`empty_like` 实测 **1.6 µs**；
`DataPtr` 析构只把块还回池子、不碰驱动。本机 9.97 → **8.77 ms**，跨进程 spread 12.2% → 0.5%。

### 第三代：file-static `cudaMalloc`、grow-only（`3711d83`）

LeetGPU 不允许 `#include <torch/extension.h>`，所以临时改成：

```cpp
static float* g_scratch = nullptr;
static int64_t g_scratch_elems = 0;
if (scratch_elems > g_scratch_elems) {
    if (g_scratch) CHECK_CUDA_ERROR(cudaFree(g_scratch));
    CHECK_CUDA_ERROR(cudaMalloc(&g_scratch, scratch_elems * sizeof(float)));
    g_scratch_elems = scratch_elems;
}
```

它在打榜场景下合理（进程只跑一个形状、单线程串行），但**不适合搬进推理框架**：
非线程安全、非多 stream / 多 device 安全、对框架的内存规划不可见、进程存活期间一直占着，
且"释放"只发生在形状变大时（占用是见过的最大值，不是累加）。

### 第四代：回到 caching allocator（`823b516`，当前版本）

不再打榜后，把第三代换回第二代，并补上两件只有接框架时才会暴露的事：

```cpp
const c10::cuda::CUDAGuard device_guard(Q.device());   // 入口：分配跟着 Q 的设备走

torch::Tensor scratch = torch::empty(
    {scratch_elems},
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

flash_decoding_chunked_attention(..., at::cuda::getCurrentCUDAStream());  // 两次 launch 都用当前流
```

- **设备**：`torch::empty` 按*当前*设备分配，而 Q/K/V 可能刚在另一个设备上创建；
  `CUDAGuard` 把当前设备钉到 Q 的设备，保证三者一致。
- **流**：原来 `<<<grid, block>>>` 走 legacy default stream，而 torch 的算子在它自己的 current stream 上；
  两者不同流时，allocator 回收的时序与 kernel 执行没有顺序保证。改成显式传 `getCurrentCUDAStream()`
  之后，workspace、输入、输出在同一条流上排队。

成本：每次调用多一次 `torch::empty`（约 1.6 µs）+ 一次 device guard（纳秒级），端到端仍在 **8.77 ms**。

### 给推理框架的建议

三种正经做法，按侵入性从低到高：

1. **调用方传入 workspace**（推荐）：把 `flash_decoding_attention_cuda` 扩成接受 `void* workspace, size_t bytes`，
   由框架在模型初始化时按 `max_seq_len / max_heads` 一次性申请并持有。这样显存预算可见、生命周期明确，
   也天然支持多 stream / 多 device。
2. **用框架自己的 allocator 每次分配**（本仓库现在的做法）：对 PyTorch 来说就是 `at::empty`，
   命中 free-list 约 1.6 µs，线程安全、计入预算、可被 `torch.cuda.empty_cache()` 回收。
   只要单次调用耗时远大于 1.6 µs（本 kernel 是 8.8 ms），这点开销可以忽略。
3. **按模型配置静态预分配**：如果坚持"一次申请、永不释放"，请按 `max_seq_len` 而不是契约上限算容量——
   契约上限是 **1.004 GiB**，这在一个 16 GB 的 T4 上会白白吃掉 6.5% 显存；而按 T=64,S=8192 只要 64.5 MiB。

---

## 8. 数字汇总

| 阶段 | commit | T4 Runtime | 本机 4060 median | 关键改动 |
|---|---|---|---|---|
| 朴素两趟 | `36b88aa` | 14.8822 ms | 10.566 ms | chunk kernel + LSE reduce |
| D=128 快速路径 | `8d9422c` | **10.56 ms** | ~8.8 ms | 一 warp 一行、`float4`、4 行预取、Q 常驻寄存器 |
| head-major 块序 | `965ced7` | 10.56 ms | ~11.8 ms（降频态） | `grid.x = head`（无收益，保留） |
| 一 warp 一 head ×4 | `cff9277` | **10.3947 ms** | ~11.8 ms（降频态） | 2 KB 连续、零 `__syncthreads`、块数 -75% |
| online softmax 单趟 | `0707662` | **10.3695 ms** | 8.73–8.79 ms | K/V 同迭代、分数不进 shared |
| caching allocator + torch 流 | `823b516` | —（未再提交） | 8.772 / 8.789 ms | workspace 生命周期、`getCurrentCUDAStream` |

换算：

- T4：`2.147 GB / 10.3695 ms = 207 GB/s = 320 GB/s 的 65%`（改动前 144 GB/s = 45%；当时榜一 202 GB/s）
- 本机：`2.147 GB / 8.772 ms = 245 GB/s`，ncu 报 DRAM 97.5%（ncu 口径峰值 255.8 GB/s）
- 本机 `harness check`：`max|diff| = 3.68357e-05`（atol/rtol 1e-3）

---

## 9. 为什么停在这里

- **必读字节不可压缩**：fp32 契约下 K/V 就是 2.147 GB，T4 的物理下限是 6.71 ms。
- **三条独立实现撞在同一面墙**：我们 207 GB/s、当时的榜一 202 GB/s、Flops 视角也指向 ~200 GB/s，
  说明 65% 峰值更像平台的实际上限，而不是某个实现的缺陷。
- **剩下能动的都是零头**（每项都实测或算过）：
  - 单 kernel 融合掉 partial 的写+读（16 MiB / 2160 MiB ≈ 0.7%），且与「一 warp 一 head」的完成度统计耦合，改造成本高；
  - `O_chunk` 写加 `st.global.cs` / L2 policy ≤ 0.4%；
  - `CHUNK_SIZE` 128→512 省 0.7% 流量，但块数 16384→4096 后 40 SM 上只剩 ~8.5 波次，波尾风险 5%，否决；
  - 双缓冲 / `cp.async` 无用：按 Little 定律需求 `248 GB/s × ~500 ns ≈ 124 KB` 在途，
    而现有配置已有 24 SM × 48 warp × 32 lane × 16 B = 590 KB（通用路径 147 KB），
    这正好解释了 96.85% → 97.47% 只涨 0.6 个点。

---

## 10. 复现

```bash
cd /home/houlai/resumeProject/hpc/leetgpu
python3 setup.py build_ext --inplace          # 需要 Full access（AGENTS.md 家规 4）
python3 -m harness check flash_decoding_attention
python3 -m harness run flash_decoding_attention --procs 3
sudo ncu --set full -s 10 -c 2 python3 -m harness run flash_decoding_attention --ncu
```

相关 commit（`git log --oneline --follow -- src/flash_decoding_attention.cu`）：

```
823b516 refactor: allocate the workspace with the caching allocator
0707662 perf: online softmax single pass over K and V
cff9277 perf: one head per warp, four heads per block
965ced7 perf: head-major block order for better DRAM locality
3711d83 perf: reuse the workspace across calls without torch
8d9422c perf: vectorized D=128 path with 4 loads in flight
cad05ba perf: allocate scratch from the caching allocator
36b88aa feat: add flash decoding attention kernel (#1)
```
