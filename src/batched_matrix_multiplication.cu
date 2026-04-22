#include "utils.cuh"

#include <cuda.h>

/*
Implement a batched matrix multiplication in FP32. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
All matrices are stored in row-major order and use 32-bit floating point numbers (FP32).

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the C array
Example 1:
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
Constraints
1 ≤ B ≤ 128
1 ≤ M, N, K ≤ 1024
*/

static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


template <int BM, int BN, int BK>
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) { 
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    const float *a_ptr = A + blockIdx.z * M * K;
    const float *b_ptr = B + blockIdx.z * K * N;
    float *c_ptr = C + blockIdx.z * M * N;

    int tx = threadIdx.x % (BN);
    int ty = threadIdx.x / (BN);
    int row_c = blockIdx.y * BM + ty;
    int col_c = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int i = 0; i < K; i += BK) {
        // load a from global to shared
        for (int j = threadIdx.x; j < BM * BK; j += blockDim.x) {
            int row_a = j / BK;
            int col_a = j % BK;
            if (row_a + blockIdx.y * BM < M && col_a + i < K) {
                sA[j] = a_ptr[(blockIdx.y * BM + row_a) * K + (i + col_a)];
            } else {
                sA[j] = 0.0f;
            }
        }

        for (int j = threadIdx.x; j < BK * BN; j += blockDim.x) {
            int row_b = j / BN;
            int col_b = j % BN;
            if (row_b + i < K && col_b + blockIdx.x * BN < N) {
                sB[j] = b_ptr[(i + row_b) * N + (blockIdx.x * BN + col_b)];
            } else {
                sB[j] = 0.0f;
            }
        }
        __syncthreads();

        // compute
        if (row_c < M && col_c < N) {
            for (int k = 0; k < BK; ++k) {
                acc += sA[ty * BK + k] * sB[k * BN + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        c_ptr[row_c * N + col_c] = acc;
    }
}


void batched_matmul_naive_cuda(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 threads(BN * BM);
    dim3 blocks(div_up(N, BN), div_up(M, BM), BATCH);
    batched_matmul_kernel<BM, BN, BK><<<blocks, threads>>>(A, B, C, BATCH, M, N, K);
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
 * 矩阵乘法 C = A * B
    @param A, B, C are device pointers, A[BATCH, M, K], B[BATCH, K, N], C[BATCH, M, N]
 */
static __global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) 
{
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const float *a_ptr = A + blockIdx.z * M * K;
    const float *b_ptr = B + blockIdx.z * K * N;
    float *c_ptr = C + blockIdx.z * M * N;

    // 2. 共享内存声明
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // 双缓冲 A 矩阵，大小为 8 x 128，转置存储
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // 双缓冲 B 矩阵，大小为 8 x 128

    // 3. 线程映射 id 计算
    // copy A 按 8列 x 32行的方式映射线程
    int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 (对应 K 维度)
    int tAy = threadIdx.x / THREAD_A_LAYOUT_X;       // 0-31 (对应 M 维度)

    // copy B 按 32列 x 8行 的方式映射线程
    int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-31 (对应 K 维度)
    int tBy = threadIdx.x / THREAD_B_LAYOUT_X;       // 0-7 (对应 N 维度)

    // C 矩阵计算的线程映射：每个线程负责计算 C 中一个 8x8 的子块
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // 0-3
    // int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    // int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3
    int lanex = laneId & 7; // 0-7
    int laney = laneId >> 3; // 0-3
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // printf("threadIdx: %d, warpId: %d, tAx: %d, tAy: %d, tBx: %d, tBy: %d, tCx: %d, tCy: %d\n", threadIdx.x, warpId, tAx, tAy, tBx, tBy, tCx, tCy);

    // 4. 寄存器声明
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f}; // 每个线程负责计算的 C 子块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵的寄存器缓存，双缓冲，每次加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵的寄存器缓存，双缓冲，每次加载 8 列

    int bufferId = 0; // 双缓冲切换标志，0 或 1

    // 5. 预加载 K= 0 的数据到共享内存
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
        int r = row0 + j + tAy; // M 维度的行索引
        int c = tAx; // K 维度的列索引
        sA[0][tAx][(j + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
    }
    

    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_Y) {
        int r = i + tBy; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx; // N 维度的列  
            sB[0][i + tBy][(j + tBx)] = (r < K && c < N) ? b_ptr[r * N + c] : 0.0f;
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
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 3; // 计算行偏移
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ (tk << 2)]);
                }

                // 从共享内存加载 B 矩阵到寄存器，每次加载连续的 8 个 float，有bank conflicts\
                // 由于线程间的访问步长是 8，而 Bank 数量是 32，它们的最小公倍数导致了每 4 个线程（32 / 8 = 4）就会回到同一个 Bank。
                // 这意味着：
                // 线程 0, 4, 8, 12, 16, 20, 24, 28 都在抢 Bank 0
                // 线程 1, 5, 9, 13, 17, 21, 25, 29 都在抢 Bank 8
                // #pragma unroll
                // for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                //     int c = (tCx + tn * THREAD_C_LAYOUT_X) << 3; // 计算列偏移
                //     FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c]);
                //     FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][c + 4]);
                // }
                // 从共享内存加载 B 矩阵到寄存器，每次加载 1 个 float，无 bank conflicts
                // #pragma unroll
                // for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                //     tCsB[tk & 1][tn] = sB[bufferId][tk][tn * THREAD_C_LAYOUT_X + tCx];
                // }
                // 从共享内存加载 B 矩阵到寄存器，每次加载 4 个 float，无 bank conflicts
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2; // 计算列偏移
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
                // 从共享内存加载 B 矩阵到寄存器，每次加载 2 个 float，无 bank conflicts
                // #pragma unroll
                // for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 1; tn++) {
                //     int c = (tCx + tn * THREAD_C_LAYOUT_X) << 1; // 计算列偏移
                //     FLOAT2(tCsB[tk & 1][tn << 1]) = FLOAT2(sB[bufferId][tk][c]);
                // }
            }
        }

        // 预加载下一块 K 的数据到共享内存
        if (k < K) {
            // 加载 A 矩阵的下一块数据
            #pragma unroll
            for (int j = 0; j < BLOCK_M; j+= THREAD_A_LAYOUT_Y) {
                int r = row0 + j + tAy; // M 维度的行索引
                int c = k + tAx; // K 维度的列索引
                sA[bufferId ^ 1][tAx][(j + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
            }
            

            // 加载 B 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx; // N 维度的列  
                    sB[bufferId ^ 1][i + tBy][(j + tBx)] = (r < K && c < N) ? b_ptr[r * N + c] : 0.0f;
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
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3); // C 的列索引，配合寄存器加载的列偏移，对应于一次加载 float4
            // int c = col0 + tCx * THREAD_C_X_TILE_SIZE + j; // C 的列索引，配合寄存器加载的列偏移，对应于一次加载 float
            // int c = col0 + (tCx << 1) + (j & (~1)) * THREAD_C_LAYOUT_X + (j & 1); // C 的列索引，配合寄存器加载的列偏移, 对应于一次加载 float2
            // int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7); // C 的列索引，配合寄存器加载的列偏移, 对应于一次加载 float8
            if (r < M && c < N) {
                c_ptr[r * N + c] = acc[i][j];
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
static void launch_batched_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) 
{
    printf("launch_batched_matmul_kernel with BATCH=%d, M=%d, N=%d, K=%d\n", BATCH, M, N, K);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_up(N, BLOCK_N), div_up(M, BLOCK_M), BATCH);
    batched_matmul_kernel<<<gridDim, blockDim>>>(A, B, C, BATCH, M, N, K);
}


void batched_matmul_cuda(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    launch_batched_matmul_kernel(A, B, C, BATCH, M, N, K);
} 