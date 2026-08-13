#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>



namespace mhca_matmul_pipeline {

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
 * @brief    QK multiplication, scale, and add delta
 * @param    A, B, C are device pointers, A[batch, M, d], B[batch, N, d], C[batch, M, N]
 * @param    scale is the scaling factor for the dot product, typically 1/sqrt(d)
 * @return
 */
__global__ void batched_matmul_rc_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale, int batch_size) 
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const float *a_ptr = A + blockIdx.z * M * K;
    const float *b_ptr = B + blockIdx.z * N * K;
    float *c_ptr = C + blockIdx.z * M * N;

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
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
        }
    }
    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
        int r = col0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
            int c = 0 + j + tBx;
            sB[0][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? b_ptr[r * K + c] : 0.0f;
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
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
                int r = col0 + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
                    int c = k + j + tBx;
                    sB[bufferId ^ 1][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? b_ptr[r * K + c] : 0.0f;
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
                c_ptr[r * N + c] = acc[i][j] * scale;
            }
        }
    }
}


void launch_batched_matmul_rc_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale, int batch_size) 
{
    // printf("launch_batched_matmul_rc_pipeline_scale_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M), batch_size);
    batched_matmul_rc_pipeline_scale_kernel<<<grid, block>>>(A, B, C, M, N, K, scale, batch_size);
}


/**
 * @brief    Transpose QKV from [seq_len, h, size_per_head] to [h, seq_len, size_per_head]
 * @param    Q [M, h, size_per_head]
 * @param    K [N, h, size_per_head]
 * @param    V [N, h, size_per_head]
 * @param    Q_trans [h, M, size_per_head]
 * @param    K_trans [h, N, size_per_head]
 * @param    V_trans [h, N, size_per_head]
 */
__global__ void transpose_qkv_kernel(
    const float * __restrict__ Q, 
    const float * __restrict__ K, 
    const float * __restrict__ V, 
    float * __restrict__ Q_trans, 
    float * __restrict__ K_trans, 
    float * __restrict__ V_trans, 
    int M, int N, int num_heads, int size_per_head)
{
    const int head_id = blockIdx.x;
    const int id_in_head = threadIdx.x;

    int seq_idx;
    int src_idx;
    int dst_idx;

    // trans Q from [M, h, size_per_head] to [h, M, size_per_head]
    if (blockIdx.y < M) {
        seq_idx = blockIdx.y;
        src_idx = seq_idx * (num_heads * size_per_head) + head_id * size_per_head + id_in_head;
        dst_idx = head_id * (M * size_per_head) + seq_idx * size_per_head + id_in_head;
        Q_trans[dst_idx] = Q[src_idx];
    }
    // trans K and V from [N, h, size_per_head] to [h, N, size_per_head]
    else if (blockIdx.y < (M + N)) {
        seq_idx = blockIdx.y - M;
        src_idx = seq_idx * (num_heads * size_per_head) + head_id * size_per_head + id_in_head;
        dst_idx = head_id * (N * size_per_head) + seq_idx * size_per_head + id_in_head;
        K_trans[dst_idx] = K[src_idx];
        V_trans[dst_idx] = V[src_idx];
    }
}


void launch_transpose_qkv_kernel(
    const float * Q, const float * K, const float * V, 
    float * Q_trans, float * K_trans, float * V_trans, 
    int M, int N, int num_heads, int size_per_head)
{
    dim3 block(size_per_head);
    dim3 grid(num_heads, M + N);
    transpose_qkv_kernel<<<grid, block>>>(Q, K, V, Q_trans, K_trans, V_trans, M, N, num_heads, size_per_head);
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[h, N, N], attn_scores[h, N, N]
 * @param    N sequence length
 * @return   void
 */
__global__ void softmax_kernel(const float* qk, float* attn_scores, int N) 
{
    float max_val = -INFINITY;
    float exp_sum = 0.0f;
    float val;

    // Find max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    // Compute exp sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        exp_sum += expf(val - max_val);
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    // Write output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        attn_scores[blockIdx.x * N + i] = expf(val - max_val) / exp_sum;
    }
}



/**
 * 批量矩阵乘法+转置
 * @param A, B, C are device pointers, A[batch, M, d], B[batch, N, d], C[M, batch, N]
 */
static __global__ void batched_matmul_pipeline_transpose_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, int batch_size)
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const float* A_ptr = A + blockIdx.z * M * K;
    const float* B_ptr = B + blockIdx.z * N * K;

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
            sA[0][j + tAx][(i + tAy) ^ (tAx << 2)] = (r < M && c < K) ? A_ptr[r * K + c] : 0.0f;
        }
    }

    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B_ptr[r * N + c] : 0.0f;
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
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A_ptr[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B_ptr[r * N + c] : 0.0f;
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
                // 转置后的索引
                int dist_idx = r * batch_size * N + blockIdx.z * N + c;
                C[dist_idx] = acc[i][j];
            }
        }
    }
}


static void launch_batched_matmul_pipeline_transpose_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, int batch_size) 
{
    // printf("launch_batched_matmul_pipeline_transpose_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M), batch_size);
    batched_matmul_pipeline_transpose_kernel<<<grid, block>>>(A, B, C, M, N, K, batch_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



// Q, K, V, output are device pointers
/**
 * @brief    Multi-head attention using pipelined matrix multiplication for QK and AV multiplication
 * @param    Q, K, V, output are device pointers, 
 * @param    Q[N, h, size_per_head], K[N, h, size_per_head], V[N, h, size_per_head], output[N, h, size_per_head]
 */
void mhca_matmul_pipeline_cuda(const float* Q, const float* K, const float* V, float* output, 
    int M, int N, int H, int D) 
{
    printf("mhca_matmul_pipeline_cuda\n");
    // qk and scaled
    float* buf;
    int buf_size = M * H * D * sizeof(float) + 2 * N * H * D * sizeof(float) + H * M * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&buf, buf_size));
    float* Q_trans = buf;
    float* K_trans = Q_trans + M * H * D;
    float* V_trans = K_trans + N * H * D;
    float* qk = V_trans + N * H * D;
    float* attn_scores = qk + H * M * N;

    // transpose Q\K\V to [h, N/M, size_per_head]
    launch_transpose_qkv_kernel(Q, K, V, Q_trans, K_trans, V_trans, M, N, H, D);

    // QK^T / sqrt(d_model)
    float scale = 1.0f / sqrtf((float)D);
    launch_batched_matmul_rc_pipeline_scale_kernel(Q_trans, K_trans, qk, M, N, D, scale, H);

    // Softmax
    softmax_kernel<<<H * M, 256>>>(qk, attn_scores, N);

    // attn_scores * V
    // attn_scores[h, M, N] @ V[h, N, D] -> [h, M, D] -> attn_out[M, h, D]
    launch_batched_matmul_pipeline_transpose_kernel(attn_scores, V_trans, output, M, D, N, H);

    CHECK_CUDA_ERROR(cudaFree(buf)); 
}

} // namespace mhca_matmul_pipeline



torch::Tensor multi_head_cross_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.dim() == 3, "Q must be 3-dimensional");
    TORCH_CHECK(K.dim() == 3, "K must be 3-dimensional");
    TORCH_CHECK(V.dim() == 3, "V must be 3-dimensional");

    int M = Q.size(0);
    int N = K.size(0);
    int H = Q.size(1);
    int D = Q.size(2);
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of heads");
    TORCH_CHECK(Q.size(2) == K.size(2), "Q and K must have the same size_per_head");
    TORCH_CHECK(Q.size(2) == V.size(2), "Q and V must have the same size_per_head");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of heads");

    torch::Tensor output = torch::zeros({M, H, D}, Q.options());
    mhca_matmul_pipeline::mhca_matmul_pipeline_cuda(
        Q.data_ptr<float>(), 
        K.data_ptr<float>(), 
        V.data_ptr<float>(), 
        output.data_ptr<float>(), 
        M, N, H, D);

    return output;
}


