#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>

namespace {
    constexpr int TILE_SIZE = 16;

}


namespace mha_naive {

/**
 * @brief    Multi-head QK scaled
 * @param    Q, K, qk are device pointers, Q[N, h, size_per_head], K[N, h, size_per_head], qk[h, N, N]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void multi_head_qk_scaled_kernel(const float* Q, const float*K, float* qk, int N, int size_per_head, int d_model) 
{
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int offset_c = head_id * N * N;
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < size_per_head; k += TILE_SIZE) {
        // load Q and K to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_q[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? Q[row * d_model + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_k[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? K[row * d_model + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < N) {
        qk[offset_c + row_c * N + col_c] = acc / sqrtf((float)size_per_head);
    }
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
 * @brief    Multi-head attention
 * @param    attn_scores, V, output are device pointers, attn_scores[h, N, N], V[N, h, size_per_head], output[N, h, size_per_head]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void mha_kernel(const float* attn_scores, const float* V, float* output, int N, int size_per_head, int d_model) 
{ 
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_attn[TILE_SIZE * TILE_SIZE];
    __shared__ float s_v[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        // load attn_scores and V to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_attn[i] = (row < N && col < N) ? attn_scores[head_id * N * N + row * N + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_v[i] = (row < N && col < size_per_head) ? V[row * d_model + head_id * size_per_head + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < size_per_head) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_attn[ty * TILE_SIZE + i] * s_v[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < size_per_head) {
        output[row_c * d_model + col_c + head_id * size_per_head] = acc;
    }
}




// Q, K, V, output are device pointers
void mha_naive_cuda(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) 
{
    int size_per_head = d_model / h;
    // qk and scaled
    float* buf;
    int buf_size = h * N * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&buf, buf_size));
    float* qk = buf;
    float* attn_scores = qk + h * N * N;

    // QK^T / sqrt(d_model)
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE), h);
    multi_head_qk_scaled_kernel<<<grid, block>>>(Q, K, qk, N, size_per_head, d_model);

    // Softmax
    softmax_kernel<<<h * N, 256>>>(qk, attn_scores, N);

    // attn_scores * V
    grid = dim3(div_up(size_per_head, TILE_SIZE), div_up(N, TILE_SIZE), h);
    mha_kernel<<<grid, block>>>(attn_scores, V, output, N, size_per_head, d_model);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(buf)); 
}

} // namespace mha_naive


namespace mha_matmul_pipeline {

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
 * @brief    Transpose QKV from [N, h, size_per_head] to [h, N, size_per_head]
 */
__global__ void transpose_qkv_kernel(
    const float * Q, const float * K, const float * V, 
    float * Q_trans, float * K_trans, float * V_trans, 
    int N, int num_heads, int size_per_head)
{
    const int qkv_id = blockIdx.z; // 0: Q, 1: K, 2: V
    const int head_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    const int id_in_head = threadIdx.x;

    const float* src = (qkv_id == 0) ? Q : ((qkv_id == 1) ? K : V);
    float* dst = (qkv_id == 0) ? Q_trans : ((qkv_id == 1) ? K_trans : V_trans);

    int src_idx = seq_id * (num_heads * size_per_head) + head_id * size_per_head + id_in_head;
    int dst_idx = head_id * (N * size_per_head) + seq_id * size_per_head + id_in_head;
    dst[dst_idx] = src[src_idx];
}


void launch_transpose_qkv_kernel(
    const float * Q, const float * K, const float * V, 
    float * Q_trans, float * K_trans, float * V_trans, 
    int N, int num_heads, int size_per_head)
{
    dim3 block(size_per_head);
    dim3 grid(num_heads, N, 3);
    transpose_qkv_kernel<<<grid, block>>>(Q, K, V, Q_trans, K_trans, V_trans, N, num_heads, size_per_head);
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
void mha_matmul_pipeline_cuda(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) 
{
    printf("mha_matmul_pipeline_cuda\n");
    int size_per_head = d_model / h;
    // qk and scaled
    float* buf;
    int buf_size = N * d_model * 3 * sizeof(float) + h * N * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&buf, buf_size));
    float* Q_trans = buf;
    float* K_trans = Q_trans + N * d_model;
    float* V_trans = K_trans + N * d_model;
    float* qk = V_trans + N * d_model;
    float* attn_scores = qk + h * N * N;

    // transpose Q\K\V to [h, N, size_per_head]
    launch_transpose_qkv_kernel(Q, K, V, Q_trans, K_trans, V_trans, N, h, size_per_head);

    // QK^T / sqrt(d_model)
    float scale = 1.0f / sqrtf((float)size_per_head);
    launch_batched_matmul_rc_pipeline_scale_kernel(Q_trans, K_trans, qk, N, N, size_per_head, scale, h);

    // Softmax
    mha_naive::softmax_kernel<<<h * N, 256>>>(qk, attn_scores, N);

    // attn_scores * V
    // attn_scores[h, N, N] @ V[h, N, SIZE_PER_HEAD] -> [h, N, SIZE_PER_HEAD] -> attn_out[N, h, SIZE_PER_HEAD]
    launch_batched_matmul_pipeline_transpose_kernel(attn_scores, V_trans, output, N, size_per_head, N, h);

    CHECK_CUDA_ERROR(cudaFree(buf)); 
}

} // namespace mha_matmul_pipeline



// Q, K, V, output are device pointers
void mha_cuda(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) 
{
    // mha_naive::mha_naive_cuda(Q, K, V, output, N, d_model, h);
    mha_matmul_pipeline::mha_matmul_pipeline_cuda(Q, K, V, output, N, d_model, h);
}


torch::Tensor mha(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int heads
) { 
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");
    TORCH_CHECK(Q.size(1) % heads == 0, "Q size must be divisible by heads");

    int N = Q.size(0);
    int d_model = Q.size(1);
    
    torch::Tensor output = torch::zeros({N, d_model}, Q.options());
    torch::Tensor attn_scores = torch::zeros({heads, N, N}, Q.options());
    mha_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        N, d_model, heads
    );

    return output;
}