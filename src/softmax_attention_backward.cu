#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>




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

// A 矩阵加载时的线程布局：32列 x 8行
constexpr int THREAD_A_T_LAYOUT_X = 32;
constexpr int THREAD_A_T_LAYOUT_Y = BLOCK_SIZE / THREAD_A_T_LAYOUT_X; // 8

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
 * @brief    QK multiplication
 * @param    A, B, C are device pointers, A[M, d], B[N, d], C[M, N]
 * @return
 */
static __global__ void matmul_rc_pipeline_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) 
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
                C[r * N + c] = acc[i][j];
            }
        }
    }
}


static void launch_matmul_rc_pipeline_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) 
{
    printf("launch_matmul_rc_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_pipeline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}




/**
 * 矩阵乘法 C = A * B，使用共享内存进行分块矩阵乘法，并且实现了双缓冲和预加载机制，以隐藏内存访问延迟。
 * @param A, B, C are device pointers, A[K, M], B[K, N], C[M, N]
 */
static __global__ void matmul_cr_pipeline_kernel(const float* A, const float* B, float* C, int M, int N, int K) { 
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 分配共享内存，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：32列 x 8行
    const int tAx = threadIdx.x & (THREAD_A_T_LAYOUT_X - 1); // 0-31 列索引
    const int tAy = threadIdx.x / THREAD_A_T_LAYOUT_X; // 0-7 行索引
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
    for (int i = 0; i < BLOCK_K; i += THREAD_A_T_LAYOUT_Y) {
        int r = 0 + i + tAy;
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j += THREAD_A_T_LAYOUT_X) {
            int c = row0 + j + tAx;
            sA[0][i + tAy][j + tAx] = (r < K && c < M) ? A[r * M + c] : 0.0f;
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
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4)]);
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
            for (int i = 0; i < BLOCK_K; i += THREAD_A_T_LAYOUT_Y) {
                int r = k + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_M; j += THREAD_A_T_LAYOUT_X) {
                    int c = row0 + j + tAx;
                    sA[bufferId ^ 1][i + tAy][j + tAx] = (r < K && c < M) ? A[r * M + c] : 0.0f;
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




/**
 * gemm kernel  NT
 * @param A: [K, M]
 * @param B: [K, N]
 * @param C: [M, N]
 */
static void launch_matmul_cr_pipeline_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) 
{
    printf("launch_matmul_cr_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_cr_pipeline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}





static __global__ void scale_softmax_kernel(
    const float * __restrict__ input,
    float * __restrict__ output, 
    int N, float scale)
{
    const float* ptr_inp = input + blockIdx.x * N;
    float* ptr_out = output + blockIdx.x * N;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        max_val = max(max_val, ptr_inp[i] * scale);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    float exp_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        exp_sum += expf(ptr_inp[i] * scale - max_val);
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        ptr_out[i] = expf(ptr_inp[i] * scale - max_val) / exp_sum;
    }
}


/**
 * @brief    scale and Softmax
 * @param    input [M, N]
 * @param    output [M, N]
 */
void launch_scale_softmax_kernel(const float* input, float *output, int M, int N, float scale) {
    // 1. 定义 block 和 grid
    dim3 block(256); // 每个 block 256 个线程
    dim3 grid(M); // 每个 block 处理一行，256 列为一个 block

    // 2. 调用 kernel
    scale_softmax_kernel<<<grid, block>>>(input, output, N, scale);
}



/**
 * @brief    Compute dS (Softmax Backward)
 * @param    attn_weights [M, N]
 * @param    dP [M, N]
 * @param    dS [M, N]
 */
static __global__ void compute_dS_kernel(
    const float * __restrict__ attn_weights,
    const float * __restrict__ dP,
    float * __restrict__ dS,
    int N, float scale
) {
    const float* ptr_attn_weights = attn_weights + blockIdx.x * N;
    const float* ptr_dP = dP + blockIdx.x * N;
    float* ptr_dS = dS + blockIdx.x * N;

    // 先逐元素相乘，再沿列(dim=1)求和，保持维度以便广播
    // shape: [M, N] -> [M, 1]
    // row_sums = (attn_weights * dP).sum(dim=1, keepdim=True)

    float row_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        row_sum += ptr_attn_weights[i] * ptr_dP[i];
    }
    __syncthreads();
    row_sum = block_all_reduce_sum(row_sum);

    // dP - row_sums 会利用广播机制，将 [M, 1] 减到 [M, N] 的每一列
    // dS = attn_weights * (dP - row_sums)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        ptr_dS[i] = ptr_attn_weights[i] * (ptr_dP[i] - row_sum) * scale;
    }
}


static void launch_compute_dS_kernel(const float* attn_weights, const float* dP, 
    float* dS, int M, int N, float scale) {
    dim3 block(256); // 每个 block 256 个线程
    dim3 grid(M); // 每个 block 处理一行，256 列为一个 block
    compute_dS_kernel<<<grid, block>>>(attn_weights, dP, dS, N, scale);
}



/**
 * Softmax Attention Backward
 * @param Q: (M, d)
 * @param K: (N, d)
 * @param V: (N, d)
 * @param dO: (M, d)
 * @param dQ: (M, d)
 * @param dK: (N, d)
 * @param dV: (N, d)
 */
// Q, K, V, dO, dQ, dK, dV are device pointers
void softmax_attention_backward_cuda(const float* Q, const float* K, const float* V, 
    const float* dO, float* dQ, float* dK, float* dV, int M, int N, int d) 
{
    float* d_buf;
    size_t buf_size = M * N * sizeof(float) * 4;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, buf_size));  
    float *qk = (float*)d_buf; // QK intermediate result
    float *attn_scores = (float*)(qk + M * N); // Attention scores intermediate result
    float *dP = (float*)(attn_scores + M * N); // dP intermediate result
    float *dS = (float*)(dP + M * N); // dS intermediate result

    float scale = 1.0f / sqrtf((float)d);

    // 前向重算: 计算 Attention Weights (P)
    // S = Q @ K.T / scale  -> shape [M, N]
    // P = softmax(S)       -> shape [M, N]
    launch_matmul_rc_pipeline_kernel(Q, K, qk, M, N, d);
    launch_scale_softmax_kernel(qk, attn_scores, M, N, scale);

    // 计算 dV
    // 公式: dV = P^T @ dO
    // [M, N]^T @ [M, d] -> [N, d]
    launch_matmul_cr_pipeline_kernel(attn_scores, dO, dV, N, d, M);

    // 计算 dP
    // 公式: dP = dO @ V^T
    // [M, d] @ [N, d]^T -> [M, N]
    launch_matmul_rc_pipeline_kernel(dO, V, dP, M, N, d);

    // 计算 dS (Softmax Backward)
    //  dS_ij = P_ij * (dP_ij - sum_k(P_ik * dP_ik)) * scale
    launch_compute_dS_kernel(attn_scores, dP, dS, M, N, scale);

    // 计算 dQ
    // 公式: dQ = dS_tilde @ K
    // [M, N] @ [N, d] -> [M, d]
    launch_matmul_pipeline_kernel(dS, K, dQ, M, d, N);

    // 计算 dK
    // 公式: dK = dS_tilde^T @ Q
    // [M, N]^T @ [M, d] -> [N, d]
    launch_matmul_cr_pipeline_kernel(dS, Q, dK, N, d, M);

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> softmax_attention_backward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& dO
) {
    // 检查输入张量的设备和数据类型
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(dO.is_cuda(), "dO must be a CUDA tensor");
    TORCH_CHECK(Q.ndimension() == 2, "Q must be a 2D tensor");
    TORCH_CHECK(K.ndimension() == 2, "K must be a 2D tensor");
    TORCH_CHECK(V.ndimension() == 2, "V must be a 2D tensor");
    TORCH_CHECK(dO.ndimension() == 2, "dO must be a 2D tensor");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same feature dimension");
    TORCH_CHECK(K.size(0) == V.size(0), "K and V must have the same sequence length");
    TORCH_CHECK(Q.size(0) == dO.size(0), "Q and dO must have the same batch size");
    
    int M = Q.size(0);
    int N = K.size(0);
    int d = Q.size(1);

    // 创建输出张量 dQ, dK, dV
    auto options = Q.options();
    torch::Tensor dQ = torch::empty({M, d}, options);
    torch::Tensor dK = torch::empty({N, d}, options);
    torch::Tensor dV = torch::empty({N, d}, options);

    softmax_attention_backward_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        dO.data_ptr<float>(),
        dQ.data_ptr<float>(),
        dK.data_ptr<float>(),
        dV.data_ptr<float>(),
        M, N, d
    );

    return std::make_tuple(dQ, dK, dV);
}