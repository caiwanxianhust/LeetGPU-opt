#include "utils.cuh"
#include <cfloat>

#include "cuda_kernels.cuh"

#include <torch/extension.h>

/*
Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix V of size N×d, your program should compute the output matrix using the formula:
 
where the softmax function is applied row-wise.

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output matrix output

Constraints
Matrix Q is of size M×d and matrices K and V are of size N×d
1 ≤ M, N ≤ 100,000
1 ≤ d ≤ 1024
*/



// flash attention kernel
template <int Bc>
__global__ void flash_softmax_attention_naive_kernel(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{ 
    const int qo_offset = blockIdx.x * d; // Offset for the current query row
    extern __shared__ float s_q[];
    __shared__ float s_qk[Bc];
    __shared__ MD_F s_row_ml_prev;

    // 显式初始化共享变量
    if (threadIdx.x == 0) {
        s_row_ml_prev.m = -FLT_MAX;
        s_row_ml_prev.d = 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        s_q[i] = Q[qo_offset + i];
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        // Initialize s_qk
        for (int i = threadIdx.x; i < Bc; i += blockDim.x) {
            s_qk[i] = 0.0f;
        }
        __syncthreads();

        // Compute dot products
        for (int i = 0; i< Bc && (j + i) < N; ++i) {
            float dot = 0.0f;
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                dot += s_q[k] * K[(j + i) * d + k];
            }
            // Reduce within block
            __syncthreads();
            dot = block_all_reduce_sum(dot);
            
            if (threadIdx.x == 0) {
                s_qk[i] = dot / sqrtf((float)d);
            }
        }
        __syncthreads();

        // Compute softmax and weighted sum
        MD_F row_ml, tmp;
        row_ml.m = -FLT_MAX;
        row_ml.d = 0.0f;
        for (int i=threadIdx.x; i<Bc && (j+i)<N; i+=blockDim.x) {
            tmp.m = s_qk[i];
            tmp.d = 1.0f;
            row_ml = MDF_Op()(row_ml, tmp);
        }
        __syncthreads();
        row_ml = block_all_reduce_sum_mdf(row_ml);
        // Combine with previous blocks
        MD_F row_ml_new = MDF_Op()(s_row_ml_prev, row_ml);

        float rlf = 1.0f / row_ml_new.d;
        float exp_m_diff = expf(s_row_ml_prev.m - row_ml_new.m);
        float exp_m_curr = expf(row_ml.m - row_ml_new.m);

        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            float pv = 0.0f;
            for (int i = 0; i < Bc && (j + i) < N; ++i) {
                pv += expf(s_qk[i] - row_ml.m) * V[(j + i) * d + k];
            }
            
            output[qo_offset + k] = rlf * (exp_m_diff * s_row_ml_prev.d * output[qo_offset + k] + exp_m_curr * pv);
        }

        if (threadIdx.x == 0) {
            s_row_ml_prev = row_ml_new;
        }
        __syncthreads();
    }
}


void flash_softmax_attention_naive_cuda(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{
    int block_size = 256;
    int grid_size = M; // One block per output row
    constexpr int Bc = 256;
    int sram_size = sizeof(float) * d;

    flash_softmax_attention_naive_kernel<Bc><<<grid_size, block_size, sram_size, 0>>>(Q, K, V, output, M, N, d);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
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

}


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
    printf("launch_matmul_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_pipline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}


/**
 * @brief    QK multiplication, scale
 * @param    A, B, C are device pointers, A[M, d], B[N, d], C[M, N]
 * @param    alpha is the slope controlling the linear bias
 * @return
 */
static __global__ void matmul_rc_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale) 
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存分配，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：8列 x 32行
    const int tBx = threadIdx.x & (THREAD_B_T_LAYOUT_X - 1); // 0-7 列索引
    const int tBy = threadIdx.x / THREAD_B_T_LAYOUT_X; // 0-31 行索引
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
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }
    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
        int r = col0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
            int c = 0 + j + tBx;
            sB[0][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
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
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ ((tk << 2))]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ ((tk << 2))]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c ^ ((tk << 2))]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4) ^ ((tk << 2))]);
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
            for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
                int r = col0 + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
                    int c = k + j + tBx;
                    sB[bufferId ^ 1][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        // 切换缓冲区
        bufferId ^= 1;
    }

    // 7. 将计算结果写回全局内存，写回时进行 scale 和 bias 的处理
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r =  row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j] * scale;
            }
        }
    }
}


static void launch_matmul_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale) 
{
    printf("launch_matmul_pipeline_scale_and_bias_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_pipeline_scale_kernel<<<grid, block>>>(A, B, C, M, N, K, scale);
}


/**
 * @brief    Softmax kernel each row is processed by a warp, so blockDim.x should be set to 32 or its multiple
 * @param    qk, attn_score are device pointers, qk[M, N], attn_score[M, N]
 */
static __global__ void warp_softmax_kernel(const float * __restrict__ qk, float * __restrict__ attn_scores, int M, int N) {
    int warpId = threadIdx.x >> 5; // 0-7 warpId，32 线程一个 warp
    int laneId = threadIdx.x & 31; // 0-31 laneId
    int row = blockIdx.x * (blockDim.x >> 5) + warpId;
    if (row >= M) return;

    // 1. 找到当前行的最大值，进行数值稳定的 softmax 计算
    float max_val = -FLT_MAX;
    for (int i = laneId; i < N; i += 32) {
        float val = qk[row * N + i];
        max_val = max(max_val, val);
    }
    __syncwarp();
    // warp 内部求 max
    max_val = warp_all_reduce_max(max_val);

    // 2. 计算当前行的 exp 和 sum
    float exp_sum = 0.0f;
    for (int i = laneId; i < N; i += 32) {
        float exp_val = expf(qk[row * N + i] - max_val);
        exp_sum += exp_val;
    }
    __syncwarp();
    // warp 内部求 sum
    exp_sum = warp_all_reduce_sum(exp_sum);

    // 3. 计算 softmax 输出
    for (int i = laneId; i < N; i += 32) {
        float exp_val = expf(qk[row * N + i] - max_val);
        attn_scores[row * N + i] = exp_val / exp_sum;
    }
}


static void launch_warp_softmax_kernel(const float* qk, float* attn_score, int M, int N) {
    printf("launch_warp_softmax_kernel with M=%d, N=%d\n", M, N);
    dim3 block(BLOCK_SIZE); // 256 线程，每个 warp 32 线程，可以处理 8 行
    dim3 grid(div_up(M, BLOCK_SIZE >> 5)); // 每个 block 处理 8 行
    warp_softmax_kernel<<<grid, block>>>(qk, attn_score, M, N);
}


/**
 * @brief    Attention with Linear Biases
 * @param    Q, K, V, output are device pointers, Q[M, d], K[N, d], V[N, d], output[M, d]
 * @param    M, N, d are the dimensions of Q, K, V
 * @param    alpha is the slope controlling the linear bias
 * @return
 */
// Q, K, V, output are device pointers
static void softmax_attention_unfused_cuda(
    const float* Q, const float* K, const float* V, float* output, 
    int M, int N, int d) 
{
    float scale = 1.0f / sqrtf((float)d);

    float* buf;
    size_t buf_size = M * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&buf, buf_size));
    float *qk = buf;
    float *attn_score = buf + M * N;

    // 1. QK 乘法 + scale + add bias
    launch_matmul_pipeline_scale_kernel(Q, K, qk, M, N, d, scale);
    // 2. softmax
    launch_warp_softmax_kernel(qk, attn_score, M, N);
    // 3. attn_score 和 V 的乘法
    launch_matmul_pipeline_kernel(attn_score, V, output, M, d, N);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(buf));
}



void softmax_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{
    // flash_softmax_attention_naive_cuda(Q, K, V, output, M, N, d);
    softmax_attention_unfused_cuda(Q, K, V, output, M, N, d);
}


torch::Tensor softmax_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.ndimension() == 2, "Input Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "Input K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "Input V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Input Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "Input K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "Input V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Input Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Input K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "Input V must be of type float32");
    TORCH_CHECK(Q.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(V.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(K.size(0) == V.size(0), "Input tensors must have compatible shapes");
    int M = Q.size(0), N = K.size(0), d = Q.size(1);
    auto options = Q.options();
    torch::Tensor output = torch::empty({M, d}, options);
    softmax_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, d
    );
    return output;
}