#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cfloat>


/*
Implement Causal (masked) Self-Attention for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix V of size M×d, your program should compute the output matrix using the formula:

where mask is a causal mask that sets all positions corresponding to keys after the current query to 
.
i.e., for query i and key j:
 
The softmax function is applied row-wise. Q, K, V, and output are all of data type float32; M, and d are of data type int32.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output matrix output

Constraints
Matrix Q, K, and V are all of size M×d
1 ≤ M ≤ 10000
1 ≤ d ≤ 128
All elements in Q, K, and V are sampled from[-100.0, 100.0]
Data type for all matrices is float32
*/

namespace {
    constexpr int TILE_SIZE = 32;
    __host__ __device__ int div_up(int a, int b) {
        return (a + b - 1) / b;
    }

    #define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

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
    constexpr int THREAD_B_LAYOUT_X = 8;
    constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X; // 32

    // C 矩阵（结果）计算时的线程布局：16列 x 16行
    constexpr int THREAD_C_LAYOUT_X = 16;
    constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X; // 16

    // 每个线程负责计算的 C 矩阵分块大小
    constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_X; // 128 / 16 = 8
    constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y; // 128 / 16 = 8

    // Warp 内部细分：在 block 16列 x 16行内将 32 线程的 Warp 分为 4x8 的结构，warp 按照 4 x 2 排布
    constexpr int THREAD_C_WARP_X = 8;
    constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X; // 4
    constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X; // 2
}

/**
 * @brief    QK scaled masked
 * @param    Q, K, qk are device pointers, Q[M, d], K[M, d], qk[M, M]
 * @param    M is the number of rows
 * @param    d is the number of columns
 * @return   void
 */
static __global__ void qk_scaled_masked_kernel(const float* Q, const float* K, float* qk, int M, int d, float scale) {
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < d; k += TILE_SIZE) {
        // load data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_q[i] = (row < M && col < d) ? Q[row * d + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_k[i] = (row < M && col < d) ? K[row * d + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < M && col_c < M) { 
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < M && col_c < M) {
        qk[row_c * M + col_c] = (row_c >= col_c) ? acc * scale : -FLT_MAX;
    }
}


/**
 * 矩阵乘法 C = A * B
 * @param A 输入矩阵 A，行主序，行数 M，列数 K
 * @param B 输入矩阵 B，行主序，行数 N，列数 K（注意这里是转置的）
 * @param C 输出矩阵 C，行主序，行数 M，列数 N
 */
static __global__ void matmul_rc_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // 双缓冲 A 矩阵，大小为 8 x 128
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // 双缓冲 B 矩阵，大小为 8 x 128

    // 3. 线程映射 id 计算
    // g2s (Global to Shared) 线程映射：每个线程负责加载 A 和 B 中的一部分数据到共享内存
    int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tAy = threadIdx.x / THREAD_A_LAYOUT_X;       // 0-31 (对应 M 维度)

    int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tBy = threadIdx.x / THREAD_B_LAYOUT_X;       // 0-31 (对应 N 维度)

    // C 矩阵计算的线程映射：每个线程负责计算 C 中一个 8x8 的子块
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // 0-3
    int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // 4. 寄存器声明
    float acc[THREAD_C_X_TILE_SIZE][THREAD_C_Y_TILE_SIZE] = {0.0f}; // 每个线程负责计算的 C 子块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵的寄存器缓存，双缓冲，每次加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵的寄存器缓存，双缓冲，每次加载 8 列

    int bufferId = 0; // 双缓冲切换标志，0 或 1

    // 5. 预加载 K= 0 的数据到共享内存
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
        int c = i + tAx; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
            int r = row0 + j + tAy; // M 维度的行索引
            sA[0][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }

    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_X) {
        int c = i + tBx; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_Y) {
            int r = col0 + j + tBy; // N 维度的行索引
            sB[0][i + tBx][(j + tBy) ^ ((i + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环：细粒度流水线
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的乘积
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算 mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            // s2r 预加载下一块数据到寄存器
            if (tk < BLOCK_K) {
                // 从共享内存加载 A 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsA[tk & 1][0]) = FLOAT4(sA[bufferId][tk][ (tCy * THREAD_C_Y_TILE_SIZE) ^ (tk << 2) ]);
                FLOAT4(tCsA[tk & 1][4]) = FLOAT4(sA[bufferId][tk][(tCy * THREAD_C_Y_TILE_SIZE + 4) ^ (tk << 2)]);

                // 从共享内存加载 B 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsB[tk & 1][0]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE) ^ (tk << 2)]);
                FLOAT4(tCsB[tk & 1][4]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE + 4) ^ (tk << 2)]);
            }
        }

        // 预加载下一块 K 的数据到共享内存
        if (k < K) {
            // 加载 A 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
                int c = k + i + tAx; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
                    int r = row0 + j + tAy; // M 维度的行索引
                    sA[bufferId ^ 1][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_X) {
                int c = k + i + tBx; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_Y) {
                    int r = col0 + j + tBy; // N 维度的行索引
                    sB[bufferId ^ 1][i + tBx][(j + tBy) ^ ((i + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }
    // 7. 将计算结果写回全局内存
    // 每个线程负责写回 C 中一个 8x8 的子块
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + tCy * THREAD_C_Y_TILE_SIZE + i; // C 的行索引
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + tCx * THREAD_C_X_TILE_SIZE + j; // C 的列索引
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j];
            }
        }
    }
}


/**
 * 封装函数：用于在 host 端调用 CUDA kernel 进行矩阵乘法计算
 * A: 输入矩阵 A，形状为 (M, K)
 * B: 输入矩阵 B，形状为 (N, K)（注意这里是转置的）
 * C: 输出矩阵 C，形状为 (M, N)
 */
static void launch_matmul_rc_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    printf("Launching matmul rc kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}


/**
 * 矩阵乘法 C = A * B
 * @param A 输入矩阵 A，行主序，行数 M，列数 K
 * @param B 输入矩阵 B，行主序，行数 K，列数 N
 * @param C 输出矩阵 C，行主序，行数 M，列数 N
 */
static __global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // 双缓冲 A 矩阵，大小为 8 x 128，转置存储
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // 双缓冲 B 矩阵，大小为 8 x 128

    // 3. 线程映射 id 计算
    // copy A 按 8列 x 32行的方式映射线程
    int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tAy = threadIdx.x / THREAD_A_LAYOUT_X;       // 0-31 (对应 M 维度)

    // copy B 按 32列 x 8行 的方式映射线程
    int tBx = threadIdx.x & (32 - 1); // 0-31 (对应 K 维度)
    int tBy = threadIdx.x / 32;       // 0-7 (对应 N 维度)

    // C 矩阵计算的线程映射：每个线程负责计算 C 中一个 8x8 的子块
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // 0-3
    int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // 4. 寄存器声明
    float acc[THREAD_C_X_TILE_SIZE][THREAD_C_Y_TILE_SIZE] = {0.0f}; // 每个线程负责计算的 C 子块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵的寄存器缓存，双缓冲，每次加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵的寄存器缓存，双缓冲，每次加载 8 列

    int bufferId = 0; // 双缓冲切换标志，0 或 1

    // 5. 预加载 K= 0 的数据到共享内存
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
        int c = i + tAx; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
            int r = row0 + j + tAy; // M 维度的行索引
            sA[0][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }

    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= 8) {
        int r = i + tBy; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+= 32) {
            int c = col0 + j + tBx; // N 维度的列索引
            sB[0][i + tBy][(j + tBx)] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环：细粒度流水线
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的乘积
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算 mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            // s2r 预加载下一块数据到寄存器
            if (tk < BLOCK_K) {
                // 从共享内存加载 A 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsA[tk & 1][0]) = FLOAT4(sA[bufferId][tk][ (tCy * THREAD_C_Y_TILE_SIZE) ^ (tk << 2) ]);
                FLOAT4(tCsA[tk & 1][4]) = FLOAT4(sA[bufferId][tk][(tCy * THREAD_C_Y_TILE_SIZE + 4) ^ (tk << 2)]);

                // 从共享内存加载 B 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsB[tk & 1][0]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE)]);
                FLOAT4(tCsB[tk & 1][4]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE + 4)]);
            }
        }

        // 预加载下一块 K 的数据到共享内存
        if (k < K) {
            // 加载 A 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
                int c = k + i + tAx; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
                    int r = row0 + j + tAy; // M 维度的行索引
                    sA[bufferId ^ 1][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= 8) {
                int r = k + i + tBy; // K 维度的行索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+= 32) {
                    int c = col0 + j + tBx; // N 维度的列索引
                    sB[bufferId ^ 1][i + tBy][(j + tBx)] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }
    // 7. 将计算结果写回全局内存
    // 每个线程负责写回 C 中一个 8x8 的子块
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + tCy * THREAD_C_Y_TILE_SIZE + i; // C 的行索引
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + tCx * THREAD_C_X_TILE_SIZE + j; // C 的列索引
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j];
            }
        }
    }

}

/**
 * 封装函数：用于在 host 端调用 CUDA kernel 进行矩阵乘法计算
 * A: 输入矩阵 A，形状为 (M, K)
 * B: 输入矩阵 B，形状为 (K, N)
 * C: 输出矩阵 C，形状为 (M, N)
 */
static void launch_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) 
{
    printf("Launching matmul kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}


static __global__ void matmul_rc_scaled_masked_advanced_kernel(const float* A, const float* B, float* C, int M, int N, int K, float scale) {
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // 双缓冲 A 矩阵，大小为 8 x 128
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // 双缓冲 B 矩阵，大小为 8 x 128

    // 3. 线程映射 id 计算
    // g2s (Global to Shared) 线程映射：每个线程负责加载 A 和 B 中的一部分数据到共享内存
    int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tAy = threadIdx.x / THREAD_A_LAYOUT_X;       // 0-31 (对应 M 维度)

    int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tBy = threadIdx.x / THREAD_B_LAYOUT_X;       // 0-31 (对应 N 维度)

    // C 矩阵计算的线程映射：每个线程负责计算 C 中一个 8x8 的子块
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // 0-3
    int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // 4. 寄存器声明
    float acc[THREAD_C_X_TILE_SIZE][THREAD_C_Y_TILE_SIZE] = {0.0f}; // 每个线程负责计算的 C 子块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵的寄存器缓存，双缓冲，每次加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵的寄存器缓存，双缓冲，每次加载 8 列

    int bufferId = 0; // 双缓冲切换标志，0 或 1

    // 5. 预加载 K= 0 的数据到共享内存
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
        int c = i + tAx; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
            int r = row0 + j + tAy; // M 维度的行索引
            sA[0][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }

    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_X) {
        int c = i + tBx; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_Y) {
            int r = col0 + j + tBy; // N 维度的行索引
            sB[0][i + tBx][(j + tBy) ^ ((i + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环：细粒度流水线
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的乘积
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算 mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            // s2r 预加载下一块数据到寄存器
            if (tk < BLOCK_K) {
                // 从共享内存加载 A 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsA[tk & 1][0]) = FLOAT4(sA[bufferId][tk][ (tCy * THREAD_C_Y_TILE_SIZE) ^ (tk << 2) ]);
                FLOAT4(tCsA[tk & 1][4]) = FLOAT4(sA[bufferId][tk][(tCy * THREAD_C_Y_TILE_SIZE + 4) ^ (tk << 2)]);

                // 从共享内存加载 B 矩阵到寄存器，每次加载连续的 8 个 float
                FLOAT4(tCsB[tk & 1][0]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE) ^ (tk << 2)]);
                FLOAT4(tCsB[tk & 1][4]) = FLOAT4(sB[bufferId][tk][(tCx * THREAD_C_X_TILE_SIZE + 4) ^ (tk << 2)]);
            }
        }

        // 预加载下一块 K 的数据到共享内存
        if (k < K) {
            // 加载 A 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_A_LAYOUT_X) {
                int c = k + i + tAx; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
                    int r = row0 + j + tAy; // M 维度的行索引
                    sA[bufferId ^ 1][i + tAx][(j + tAy) ^ ((i + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_X) {
                int c = k + i + tBx; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_Y) {
                    int r = col0 + j + tBy; // N 维度的行索引
                    sB[bufferId ^ 1][i + tBx][(j + tBy) ^ ((i + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }
    // 7. 将计算结果写回全局内存
    // 每个线程负责写回 C 中一个 8x8 的子块
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + tCy * THREAD_C_Y_TILE_SIZE + i; // C 的行索引
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + tCx * THREAD_C_X_TILE_SIZE + j; // C 的列索引
            if (r < M && c < N) {
                C[r * N + c] = (r >= c) ? acc[i][j] * scale : -FLT_MAX;
            }
        }
    }
}


static void launch_qk_scaled_masked_advanced_kernel(const float* Q, const float* K, float* qk, int M, int d, float scale) {
    printf("Launching matmul_rc_scaled_masked_advanced_kernel\n");
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_up(M, BLOCK_M), div_up(M, BLOCK_N));
    matmul_rc_scaled_masked_advanced_kernel<<<gridDim, blockDim>>>(Q, K, qk, M, M, d, scale);
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[M, M], attn_scores[M, M]
 * @param    rows, cols are the dimensions of qk
 * @return   void
 */
static __global__ void softmax_kernel(const float* qk, float* attn_scores, int rows, int cols) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row_id = blockIdx.x * (blockDim.x / 32) + warp_id;

    float max_val = -FLT_MAX;
    float exp_sum = 0.0f;
    float val;

    if (row_id < rows) { 
        // find max
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = qk[row_id * cols + col_id];
            max_val = max(max_val, val);
        }
        __syncwarp();
        max_val = warp_all_reduce_max(max_val);

        // exp sum
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = exp(qk[row_id * cols + col_id] - max_val);
            exp_sum += val;
        }
        __syncwarp();
        exp_sum = warp_all_reduce_sum(exp_sum);

        // output
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = exp(qk[row_id * cols + col_id] - max_val);
            attn_scores[row_id * cols + col_id] = val / exp_sum;
        }
    }
}


/**
 * @brief    Matrix multiplication
 * @param    A, B, C are device pointers, A[M, K], B[K, N], C[M, N]
 * @param    M, N, K are the dimensions of A, B, C
 * @return   void
 */
static __global__ void matmul_fp32_kernel(const float* A, const float* B, float* C, int M, int N, int K) { 
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) { 
        // load data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_a[i] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_b[i] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < M && col_c < N) { 
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_a[ty * TILE_SIZE + i] * s_b[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = acc;
    }
}

// Q, K, V, output are device pointers
void causal_self_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float scale = 1.0f / sqrtf((float)d);

    float* d_buf;
    size_t buf_size = M * M * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, buf_size));
    float* qk = (float*)d_buf;
    float* attn_scores = qk + M * M;

    dim3 grid, block;

    // qk scaled masked
    // dim3 block(TILE_SIZE * TILE_SIZE);
    // dim3 grid(div_up(M, TILE_SIZE), div_up(M, TILE_SIZE));
    // qk_scaled_masked_kernel<<<grid, block>>>(Q, K, qk, M, d, scale);
    launch_qk_scaled_masked_advanced_kernel(Q, K, qk, M, d, scale);

    // softmax
    // each warp handles one row
    block = dim3(128);
    grid = dim3(div_up(M, 4));
    softmax_kernel<<<grid, block>>>(qk, attn_scores, M, M);

    // matmul [M, M ] @ [M, d]
    // block = dim3(TILE_SIZE * TILE_SIZE);
    // grid = dim3(div_up(d, TILE_SIZE), div_up(M, TILE_SIZE));
    // matmul_fp32_kernel<<<grid, block>>>(attn_scores, V, output, M, d, M);
    launch_matmul_kernel(attn_scores, V, output, M, d, M);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}

