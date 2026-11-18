// =============================================================================
// Multi-LoRA batched linear -- CUDA implementation
//
//   y[t] = x[t] @ W  +  (a = adapter_ids[t] >= 0)
//                        scale * (x[t] @ A[a]) @ B[a]
//
// Shapes:  x (T,K), W (K,N), A (L,K,R), B (L,R,N), ids (T,), y (T,N)
//
// Pipeline (4 kernels, all device-side, no host<->device sync between them):
//   1. bucket_kernel    : histogram + prefix-sum + scatter by adapter id.
//                         -> row_ids[t'] = original token row,
//                            gid[t']     = adapter id (O(1) group lookup),
//                            group_off[g] = start of group g in t' space.
//                         tokens with ids == -1 are excluded (base only).
//                         MAX_L = 64 covers the full constraint range.
//   2. base_gemm_tiled  : y = x @ W  (shared-tiled GEMM, OVERWRITES y)
//   3. stageA_kernel    : mid[t',r] = sum_k x[row_ids[t'],k] * A[g,k,r]
//                         (x gathered by row_ids, grouped by g, output R-wide)
//   4. stageB_kernel    : SGMV. y[row_ids[t'],n] += scale * sum_r mid[t',r]*B[g,r,n]
//                         (K dim is only R -> warp owns consecutive n, mid broadcast)
//
// Why not materialize AB = A@B ?  That turns the R-wide "narrow waist" into a
// K x N matrix: compute/token goes from R(K+N) to K*N (128x at R=16), and each
// token re-reads its own 67 MB AB[a] instead of sharing A[a]/B[a] within a group.
// An ab_baseline() is included at the bottom of the harness to measure this.
//
// Build (Windows, VS2022 + CUDA 12.8):
//   nvcc -O3 -arch=sm_89 multi_lora_cuda.cu -o multi_lora_cuda.exe
// =============================================================================

#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>


constexpr int MAX_L = 64;      // constraint: 1 <= L <= 64
constexpr int BT    = 32;      // base GEMM tile (BT x BT), block = 32x32 = 1024 thr

// -----------------------------------------------------------------------------
// 1. Bucket by adapter id (single block, 256 threads)
//    Only the INDEX is permuted; x itself is never moved.
//    Intra-group order is arbitrary (scatter uses row_ids), so shared atomicAdd
//    contention is harmless.
//    Also emits gid[t'] = adapter id, so stageA/stageB can look up the group in
//    O(1) instead of linearly scanning group_off[] (L=64 would cost ~32 probes).
// -----------------------------------------------------------------------------
__global__ void bucket_kernel(const int* __restrict__ ids,
                              int* __restrict__ row_ids,
                              int* __restrict__ gid,
                              int* __restrict__ group_off,
                              int T, int L)
{
    // 每个 group 的计数
    __shared__ int cnt[MAX_L];
    // off[g] 的可写游标副本，初值 = off[g]，phase C 里被 atomicAdd(&cur[a],1) 不断递增；循环结束时 cur[g] == off[g+1]
    __shared__ int cur[MAX_L];
    // cnt 的 exclusive prefix sum，第 g 组的起始下标，算完就不再变
    __shared__ int off[MAX_L + 1];

    const int tid = threadIdx.x;
    for (int g = tid; g < L; g += blockDim.x) cnt[g] = 0;
    __syncthreads();

    // phase A: count tokens per group
    for (int t = tid; t < T; t += blockDim.x) {
        int a = ids[t];
        if (a >= 0 && a < L) atomicAdd(&cnt[a], 1);
    }
    __syncthreads();

    // phase B: exclusive scan -> offsets, init cursors
    if (tid == 0) {
        off[0] = 0;
        for (int g = 0; g < L; ++g) {
            off[g + 1] = off[g] + cnt[g];
            cur[g]     = off[g];
        }
        //前缀和：group_off[g] = 第 g 组在 t' 空间的起始下标。
        // 第 g 组是 [group_off[g], group_off[g+1])
        for (int g = 0; g <= L; ++g) group_off[g] = off[g];
    }
    __syncthreads();

    // phase C: scatter token indices into their group's slot
    for (int t = tid; t < T; t += blockDim.x) {
        int a = ids[t];
        if (a >= 0 && a < L) {
            int p = atomicAdd(&cur[a], 1);
            // 桶内第 t' 个槽位对应原始 x 的第几行
            row_ids[p] = t;
            // 桶内第 t' 个槽位属于哪个 adapter
            gid[p]     = a;
        }
    }
}

// -----------------------------------------------------------------------------
// 2. base GEMM: y[t,n] = sum_k x[t,k]*W[k,n]   (overwrite, no init needed)
//    Plain shared-tiled GEMM. Warp reads Ws[k][tx] with consecutive tx -> coalesced.
// -----------------------------------------------------------------------------

namespace matmul_pipline {


// 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// 定义宏，用于将 float 数组强制转换为 float2 类型，实现向量化访问
#define FLOAT2(x) (reinterpret_cast<float2 *>(&(x))[0])

// --- 核心配置参数 ---
constexpr int BLOCK_M = 128;      // Block 在 M 维度（行）处理的大小
constexpr int BLOCK_N = 128;      // Block 在 N 维度（列）处理的大小
constexpr int BLOCK_K = 8;        // Block 在 K 维度（深度）处理的大小
constexpr int BLOCK_SIZE = 256;   // 每个 Block 的线程总数

// --- 线程布局配置 (这是该 Kernel 最精妙的部分) ---
// A 矩阵加载时的线程布局：8列 x 32行
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X; // 32

// B 矩阵加载时的线程布局：32列 x 8行
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X; // 8

// B 矩阵（转置）加载时的线程布局：8列 x 32行
constexpr int THREAD_B_T_LAYOUT_X = 8;
constexpr int THREAD_B_T_LAYOUT_Y = BLOCK_SIZE / THREAD_B_T_LAYOUT_X; // 32

// C 矩阵（结果）计算时的线程布局：16列 x 16行
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X; // 16

// 每个线程负责计算的 C 矩阵分块大小
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_X; // 128 / 16 = 8
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y; // 128 / 16 = 8

// Warp 内部细分：在 block 16列 x 16行内将 32 线程的 Warp 分为 8行x4列 的结构，warp 按照 4 x 2 排布
constexpr int THREAD_C_WARP_X = 4;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X; // 8
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X; // 4


/**
 * 矩阵乘法 C = A * B，使用共享内存进行分块矩阵乘法，并且实现了双缓冲和预加载机制，以隐藏内存访问延迟。
 * @param A, B, C are device pointers, A[M, K], B[K, N], C[M, N]
 */
static __global__ void matmul_pipline_kernel(const float* A, const float* B, float* C, int M, int N, int K) { 
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 分配共享内存，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：32列 x 8行
    const int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-31 列索引
    const int tBy = threadIdx.x / THREAD_B_LAYOUT_X; // 0-7 行索引
    // C 矩阵计算的线程映射，每个线程计算一个 8x8 的 C 矩阵分块
    int warpId = threadIdx.x >> 5; // 0-7 warpId，32 线程一个 warp
    int laneId = threadIdx.x & 31;
    // C 矩阵计算的 warp 布局 ：2行 x 4列
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-3
    int warpy = warpId / THREAD_C_WARP_DIM_X; // 0-1
    int lanex = laneId & (THREAD_C_WARP_X - 1); // 0-3
    int laney = laneId / THREAD_C_WARP_X; // 0-7
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15 C 矩阵列索引
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15 C 矩阵行索引

    // 4. 寄存器声明
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f}; // 每个线程计算的 C 矩阵分块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 列

    int bufferId = 0; // 双缓冲切换标识

    // 5. 预加载第一块 A 和 B 到共享内存
    // 加载 A 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
            int c = 0 + j + tAx;
            sA[0][j + tAx][(i + tAy) ^ (tAx << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }

    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的矩阵乘法
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }

            // s2r 预加载数据到寄存器
            if (tk < BLOCK_K) {
                // 加载 A 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ (tk << 2)]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个float
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4)]);
                }
            }
        }
        // 预加载 下一块 A 和 B 到共享内存
        if (k < K) {
            // 加载 A 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                    int c = k + j + tAx;
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }

    // 7. 将计算结果写回全局内存
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j];
            }
        }
    }
}


static void launch_matmul_pipeline_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // printf("launch_matmul_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_pipline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

} // namespace matmul_pipline   

// -----------------------------------------------------------------------------
// 3. stage A: mid[t',r] = sum_k x[row_ids[t'],k] * A[g,k,r]
//    One thread per (t',r) output. Warp covers consecutive r -> A[g,k,r] is
//    coalesced, and x[row_ids[t'],k] is broadcast across those lanes.
//    T_active is read from group_off[L]: grid uses T as an upper bound, so no
//    device->host copy is needed.
// -----------------------------------------------------------------------------
__global__ void stageA_kernel(const float* __restrict__ X,
                              const float* __restrict__ A,
                              const int*   __restrict__ row_ids,
                              const int*   __restrict__ gid,
                              const int*   __restrict__ group_off,
                              float* __restrict__ mid,
                              int T, int K, int R, int L)
{
    // 有效 token 数
    const int T_active = group_off[L];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_active * R) return;

    const int t = (int)(idx / R);
    const int r = (int)(idx % R);
    // t' -> original row in x
    const int orig = row_ids[t];
    // t' -> group id
    const int g = gid[t];       // O(1) group lookup (was a linear scan over L)

    // Compute mid[t', r] = sum_k x[row_ids[t'], k] * A[g, k, r]
    const float* xr = X + orig * K;
    const float* Ag = A + g * K * R;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc = fmaf(xr[k], Ag[k * R + r], acc);
    mid[t * R + r] = acc;
}

// -----------------------------------------------------------------------------
// 4. stage B (SGMV + scatter-add):
//      y[row_ids[t'],n] += scale * sum_r mid[t',r] * B[g,r,n]
//    Warp owns consecutive n -> B[g,r,n] coalesced, mid[t',r] broadcast.
//    row_ids is a permutation, so each y row is written by exactly one t':
//    plain += is race-free (base GEMM already completed in the same stream).
// -----------------------------------------------------------------------------
__global__ void stageB_kernel(const float* __restrict__ mid,
                              const float* __restrict__ B,
                              const int*   __restrict__ row_ids,
                              const int*   __restrict__ gid,
                              const int*   __restrict__ group_off,
                              float* __restrict__ Y,
                              float scale,
                              int T, int R, int N, int L)
{
    const int T_active = group_off[L];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T_active * N) return;

    const int t = (int)(idx / N);
    const int n = (int)(idx % N);

    const int g = gid[t];       // O(1) group lookup (was a linear scan over L)

    const float* Bg = B + g * R * N;
    const float* mt = mid + t * R;
    float acc = 0.0f;
    for (int r = 0; r < R; ++r)
        acc = fmaf(mt[r], Bg[r * N + n], acc);

    // t' -> original row in x
    const int orig = row_ids[t];
    Y[orig * N + n] += scale * acc;
}


// -----------------------------------------------------------------------------
// Host dispatch: 4 launches, same stream, no synchronization
// -----------------------------------------------------------------------------
void multi_lora(const float* x, const float* W, const float* A, const float* B,
    const int* adapter_ids, float* y, int T, int K, int N, int R, int L, float scale)
{
    // if (T <= 0 || K <= 0 || N <= 0 || R <= 0 || L <= 0) return;
    // if (L > MAX_L) { fprintf(stderr, "L=%d exceeds MAX_L=%d\n", L, MAX_L); exit(1); }

    const size_t b_ids = (size_t)T * sizeof(int);
    const size_t b_gid = (size_t)T * sizeof(int);
    const size_t b_off = (size_t)(L + 1) * sizeof(int);
    const size_t b_mid = (size_t)T * R * sizeof(float);
    const size_t o_ids = 0;
    const size_t o_gid = o_ids + b_ids;
    const size_t o_off = o_gid + b_gid;
    const size_t o_mid = o_off + b_off;
    const size_t total = o_mid + b_mid;

    char* base = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&base, total));

    int*   row_ids = (int*)  (base + o_ids);
    int*   gid     = (int*)  (base + o_gid);
    int*   off     = (int*)  (base + o_off);
    float* mid     = (float*)(base + o_mid);

    bucket_kernel<<<1, 256>>>(adapter_ids, row_ids, gid, off, T, L);

    matmul_pipline::launch_matmul_pipeline_kernel(x, W, y, T, N, K);

    int nA = (int)(div_up((long long)T * R, 256));
    stageA_kernel<<<nA, 256>>>(x, A, row_ids, gid, off, mid, T, K, R, L);

    int nB = (int)(div_up((long long)T * N, 256));
    stageB_kernel<<<nB, 256>>>(mid, B, row_ids, gid, off, y, scale, T, R, N, L);

    // cudaFree is an implicit device sync: it waits for all 4 async launches,
    // so releasing here is safe (no use-after-free). One alloc / one free.
    CHECK_CUDA_ERROR(cudaFree(base));
}


// x, W, A, B, adapter_ids, y are device pointers
void multi_lora_cuda(const float* x, const float* W, const float* A, const float* B,
    const int* adapter_ids, float* y, int T, int K, int N, int R, int L, float scale) 
{
    multi_lora(x, W, A, B, adapter_ids, y, T, K, N, R, L, scale);
}


torch::Tensor multi_lora_batched_linear(
    const torch::Tensor &x, 
    const torch::Tensor &W, 
    const torch::Tensor &A, 
    const torch::Tensor &B,
    const torch::Tensor &adapter_ids, 
    float scale) 
{
    int T = x.size(0);
    int K = x.size(1);
    int N = W.size(1);
    int R = A.size(2);
    int L = A.size(0);

    auto y = torch::zeros({T, N}, x.options());

    multi_lora_cuda(
        x.data_ptr<float>(), 
        W.data_ptr<float>(), 
        A.data_ptr<float>(), 
        B.data_ptr<float>(),
        adapter_ids.data_ptr<int>(), 
        y.data_ptr<float>(), 
        T, K, N, R, L, scale
    );

    return y;
}