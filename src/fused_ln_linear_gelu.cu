#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cmath>
#include <cstdint>
#include <torch/extension.h>


// var = E[d^2] - E[d]^2 with d = x - x0 (O(1) magnitude).
// (x - mean)/std == (d - E[d])/std, so this is mathematically identical
// to the textbook formula but numerically stable in float32.
static __global__ void layer_normalization_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, float eps)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int row  = blockIdx.x * (blockDim.x >> 5) + wid;
    if (row >= N) return;

    const float* x = input + row * C;
    const float x0 = x[0];

    float sum = 0.0f, sq = 0.0f;
    for (int i = lane; i < C; i += 32) {
        const float d = x[i] - x0;
        sum += d;
        sq  += d * d;
    }

    sum = warp_all_reduce_sum(sum) / (float)C;   // E[d]
    sq  = warp_all_reduce_sum(sq)  / (float)C;   // E[d^2]
    const float var = fmaxf(sq - sum * sum, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    for (int i = lane; i < C; i += 32) {
        const float d = x[i] - x0;
        const float val = (d - sum) * inv_std;
        output[row * C + i] = val * weight[i] + bias[i];
    }
}

static __global__ void layer_normalization_dp4_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, float eps)
{
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int row  = blockIdx.x * (blockDim.x >> 5) + wid;
    if (row >= N) return;

    const float* x  = input + row * C;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);
    float4* o4 = reinterpret_cast<float4*>(output + row * C);
    const int C4 = C >> 2;
    const float x0 = x[0];

    float sum = 0.0f, sq = 0.0f;
    for (int i = lane; i < C4; i += 32) {
        const float4 v = x4[i];
        const float a = v.x - x0, b = v.y - x0, c = v.z - x0, d = v.w - x0;
        sum += a + b + c + d;
        sq  += a * a + b * b + c * c + d * d;
    }

    sum = warp_all_reduce_sum(sum) / (float)C;
    sq  = warp_all_reduce_sum(sq)  / (float)C;
    const float var = fmaxf(sq - sum * sum, 0.0f);
    const float inv_std = rsqrtf(var + eps);

    for (int i = lane; i < C4; i += 32) {
        const float4 v = x4[i], w = w4[i], b = b4[i];
        float4 o;
        o.x = ((v.x - x0) - sum) * inv_std * w.x + b.x;
        o.y = ((v.y - x0) - sum) * inv_std * w.y + b.y;
        o.z = ((v.z - x0) - sum) * inv_std * w.z + b.z;
        o.w = ((v.w - x0) - sum) * inv_std * w.w + b.w;
        o4[i] = o;
    }
}

static void launch_layer_normalization_kernel(const float* input, const float* weight,
    const float* bias, float* output, int N, int C, float eps)
{
    constexpr int block_size = 128;
    const int grid_size = div_up(N, block_size >> 5);
    // float4 needs 16B alignment; cudaMalloc pointers satisfy it, but check
    // anyway in case the harness hands over a sliced/offset tensor.
    const bool aligned =
        ((((uintptr_t)input) | ((uintptr_t)weight) |
          ((uintptr_t)bias) | ((uintptr_t)output)) & 15) == 0;

    if (((C & 3) == 0) && aligned) {   // FIX: was `C & 3 == 0` (always false)
        layer_normalization_dp4_kernel<<<grid_size, block_size>>>(
            input, weight, bias, output, N, C, eps);
    } else {
        layer_normalization_kernel<<<grid_size, block_size>>>(
            input, weight, bias, output, N, C, eps);
    }
}




namespace {
// 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// 定义宏，用于将 float 数组强制转换为 float2 类型，实现向量化访问
#define FLOAT2(x) (reinterpret_cast<float2 *>(&(x))[0])

// --- 核心配置参数 ---
constexpr int BLOCK_M = 128;      // Block 在 M 维度（行）处理的大小
constexpr int BLOCK_N = 128;      // Block 在 N 维度（列）处理的大小
constexpr int BLOCK_K = 8;        // Block 在 K 维度（深度）处理的大小
constexpr int BLOCK_SIZE = 256;   // 每个 Block 的线程总数
constexpr int NWARPS = BLOCK_SIZE / 32;               // 8

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


__device__ __forceinline__ float gelu(float x) 
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118f));
}


/**
 * 矩阵乘法 C = A * B + bias，使用共享内存进行分块矩阵乘法，并且实现了双缓冲和预加载机制，以隐藏内存访问延迟。
 * @param A, B, C, bias are device pointers, A[M, K], B[K, N], C[M, N], bias[N]
 */
static __global__ void matmul_bias_gelu_pipeline_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ Bias, 
    float* __restrict__ C, 
    int M, int N, int K, bool add_bias, bool use_gelu) 
{ 
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
                float bias_val = (add_bias) ? Bias[c] : 0.0f;
                float res = acc[i][j] + bias_val;
                C[r * N + c] = (use_gelu) ? gelu(res) : res;
            }
        }
    }
}


static void launch_matmul_bias_gelu_pipeline_kernel(
    const float* A, const float* B, const float* Bias, float* C, 
    int M, int N, int K, bool add_bias, bool use_gelu) {
    // printf("launch_matmul_bias_gelu_res_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_bias_gelu_pipeline_kernel<<<grid, block>>>(A, B, Bias, C, M, N, K, add_bias, use_gelu);
}

} // namespace 


// x, gamma, beta, W, bias, y are device pointers
static void fused_ln_linear_gelu_twopass(
    const float* x, const float* gamma, const float* beta, const float* W,
    const float* bias, float* y, int M, int K, int N, float eps) 
{
    float *x_norm;
    CHECK_CUDA_ERROR(cudaMalloc(&x_norm, M * K * sizeof(float)));

    launch_layer_normalization_kernel(x, gamma, beta, x_norm, M, K, eps);

    launch_matmul_bias_gelu_pipeline_kernel(x_norm, W, bias, y, M, N, K, true, true);

    CHECK_CUDA_ERROR(cudaFree(x_norm));
}




// EXTERNAL_STATS = false : prologue computes LN statistics inside this kernel
//                          (true single-kernel fusion, but every column-tile
//                          block recomputes the same rows -- see README).
// EXTERNAL_STATS = true  : statistics come from a tiny [3, M] global table
//                          produced by ln_stats_kernel (no duplicate work).
// NORMALIZE_ON_LOAD = false : A tile is a precomputed x_norm buffer
//                             (identical to a plain GEMM, no LN work here).
template<bool EXTERNAL_STATS, bool NORMALIZE_ON_LOAD = true>
static __global__ void fused_ln_linear_gelu_kernel(
    const float* __restrict__ X,       // [M, K] raw input (NOT normalised)
    const float* __restrict__ gamma,   // [K] LN weight
    const float* __restrict__ beta,    // [K] LN bias
    const float* __restrict__ W,       // [K, N]
    const float* __restrict__ Bias,    // [N] linear bias
    float* __restrict__ Y,             // [M, N]
    const float4* __restrict__ g_stats, // [M] (x0, mean, inv_std, pad), EXTERNAL_STATS only
    int M, int N, int K, float eps)
{
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const int tid  = threadIdx.x;

    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];
    // One 128-bit load per row instead of three 32-bit loads on the hot
    // A-tile prefetch path. .x = x0, .y = E[x-x0], .z = 1/sqrt(var+eps).
    __shared__ float4 s_stats[BLOCK_M];

    const int tAx = tid & (THREAD_A_LAYOUT_X - 1);
    const int tAy = tid / THREAD_A_LAYOUT_X;
    const int tBx = tid & (THREAD_B_LAYOUT_X - 1);
    const int tBy = tid / THREAD_B_LAYOUT_X;
    const int warpId = tid >> 5, laneId = tid & 31;
    const int warpx  = warpId & (THREAD_C_WARP_DIM_X - 1);
    const int warpy  = warpId / THREAD_C_WARP_DIM_X;
    const int lanex  = laneId & (THREAD_C_WARP_X - 1);
    const int laney  = laneId / THREAD_C_WARP_X;
    const int tCx    = warpx * THREAD_C_WARP_X + lanex;
    const int tCy    = warpy * THREAD_C_WARP_Y + laney;

    // ------------------------------------------------------------------
    // PROLOGUE: row-wise shifted LayerNorm statistics (fused variants only).
    // ------------------------------------------------------------------
    if (NORMALIZE_ON_LOAD) {
    const int rows = (M - row0 < BLOCK_M) ? (M - row0) : BLOCK_M;
    if (EXTERNAL_STATS) {
        // Cheap 128-bit copy of rows floats: no reduction, no global x read.
        for (int rl = tid; rl < rows; rl += BLOCK_SIZE) {
            s_stats[rl] = g_stats[row0 + rl];
        }
    } else {
        // One warp owns one row. Vectorised loads + 2x unroll raise MLP for
        // large K; the reduction itself is the exposed serial cost.
        const bool vec4 = ((K & 3) == 0) && ((((size_t)X) & 15) == 0);
        #pragma unroll 2
        for (int rl = warpId; rl < rows; rl += NWARPS) {
            const int r = row0 + rl;
            const float* xr = X + (size_t)r * K;
            const float x0 = xr[0];                 // shift origin
            float sum = 0.0f, sq = 0.0f;
            if (vec4) {
                const float4* x4 = reinterpret_cast<const float4*>(xr);
                const int K4 = K >> 2;
                for (int k = laneId; k < K4; k += 32) {
                    const float4 v = x4[k];
                    const float a = v.x - x0, b = v.y - x0, c = v.z - x0, d = v.w - x0;
                    sum += a + b + c + d;
                    sq  += a * a + b * b + c * c + d * d;
                }
            } else {
                for (int k = laneId; k < K; k += 32) {
                    const float d = xr[k] - x0;     // O(1), Sterbenz-exact
                    sum += d;
                    sq  += d * d;
                }
            }
            sum = warp_all_reduce_sum(sum) / (float)K;  // E[d]
            sq  = warp_all_reduce_sum(sq)  / (float)K;  // E[d^2]
            const float var = fmaxf(sq - sum * sum, 0.0f);
            if (laneId == 0) {
                s_stats[rl] = make_float4(x0, sum, rsqrtf(var + eps), 0.0f);
            }
        }
    }
    __syncthreads();
    }  // NORMALIZE_ON_LOAD

    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];
    int bufferId = 0;

    // ------------------------------------------------------------------
    // Prefetch tile k=0: normalise A on load.
    // All `i` iterations of one thread share the same K column `c`, so
    // gamma[c]/beta[c] are loaded once and broadcast across 32 rows.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
        const int c = 0 + j + tAx;
        const float gam = NORMALIZE_ON_LOAD ? ((c < K) ? gamma[c] : 0.0f) : 0.0f;
        const float bet = NORMALIZE_ON_LOAD ? ((c < K) ? beta [c] : 0.0f) : 0.0f;
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
            const int rl = i + tAy;            // logical row inside tile
            const int r  = row0 + rl;
            float a = 0.0f;
            if (r < M && c < K) {
                if (NORMALIZE_ON_LOAD) {
                    const float4 st = s_stats[rl];
                    const float d = (X[(size_t)r * K + c] - st.x) - st.y;
                    a = (d * st.z) * gam + bet;
                } else {
                    a = X[(size_t)r * K + c];      // precomputed x_norm
                }
            }
            sA[0][j + tAx][rl ^ ((j + tAx) << 2)] = a;
        }
    }
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        const int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            const int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? W[(size_t)r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Mainloop: unchanged GEMM pipeline.
    // ------------------------------------------------------------------
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk < BLOCK_K) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3])       = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ (tk << 2)]);
                }
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3])       = FLOAT4(sB[bufferId][tk][c]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4)]);
                }
            }
        }

        if (k < K) {
            // Prefetch tile k: normalise A on load (same as above).
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                const int c = k + j + tAx;
                const float gam = (c < K) ? gamma[c] : 0.0f;
                const float bet = (c < K) ? beta [c] : 0.0f;
                #pragma unroll
                for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                    const int rl = i + tAy;
                    const int r  = row0 + rl;
                    float a = 0.0f;
                    if (r < M && c < K) {
                        if (NORMALIZE_ON_LOAD) {
                            const float4 st = s_stats[rl];
                            const float d = (X[(size_t)r * K + c] - st.x) - st.y;
                            a = (d * st.z) * gam + bet;
                        } else {
                            a = X[(size_t)r * K + c];   // precomputed x_norm
                        }
                    }
                    sA[bufferId ^ 1][j + tAx][rl ^ ((j + tAx) << 2)] = a;
                }
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                const int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    const int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] =
                        (r < K && c < N) ? W[(size_t)r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        bufferId ^= 1;
    }

    // ------------------------------------------------------------------
    // Epilogue: + bias, GELU, store (unchanged).
    // ------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        const int r = row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            const int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                const float res = acc[i][j] + Bias[c];
                Y[(size_t)r * N + c] = gelu(res);
            }
        }
    }
}


// ----------------------------------------------------------------------------
// single-kernel fusion. No allocation, no workspace.
// ----------------------------------------------------------------------------
// x, gamma, beta, W, bias, y are device pointers
static void fused_ln_linear_gelu_single(
    const float* x, const float* gamma, const float* beta, const float* W,
    const float* bias, float* y, int M, int K, int N, float eps) 
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    fused_ln_linear_gelu_kernel<false><<<grid, block>>>(
        x, gamma, beta, W, bias, y,
        (const float4*)nullptr, M, N, K, eps);
}


void fused_ln_linear_gelu_cuda(
    const float* x, const float* gamma, const float* beta, const float* W,
    const float* bias, float* y, int M, int K, int N, float eps) 
{
    // fused_ln_linear_gelu_twopass(x, gamma, beta, W, bias, y, M, K, N, eps);
    fused_ln_linear_gelu_single(x, gamma, beta, W, bias, y, M, K, N, eps);
}


torch::Tensor fused_ln_linear_gelu(
    const torch::Tensor &x, 
    const torch::Tensor &gamma, 
    const torch::Tensor &beta,
    const torch::Tensor &W, 
    const torch::Tensor &bias, 
    float eps) 
{

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = W.size(1);

    auto y = torch::empty({M, N}, x.options());

    fused_ln_linear_gelu_cuda(
        x.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(),
        W.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        y.data_ptr<float>(),
        M, K, N, eps);

    return y;
}