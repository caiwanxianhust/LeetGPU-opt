#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Implement Linear Attention for a given set of matrices, following the method described in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" . Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix V of size M×d, your program should compute the output matrix using the formula:
 

where 
 is a feature map applied element-wise, for example:
 
 
All matrices Q, K, V, and output are of type float32, and M and d are of type int32.

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

static __device__ __host__ __forceinline__ float phi(float x) {
    return x > 0 ? x + 1.0f : exp(x);
}



static constexpr int TILE_SIZE = 32;


__global__ void zeros_init_kernel(float* vec, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        vec[idx] = 0.0f;
    }
}


/**
 * Compute phi(Q) and phi(K)^T and sum of phi(K)
 * Q: pointer to query matrix of size Mxd
 * K: pointer to key matrix of size Mxd
 * phi_q: pointer to output phi(Q) matrix of size Mxd
 * phi_k_T: pointer to output phi(K)^T matrix of size dxM
 * ksum: pointer to output sum of phi(K) vector of size d
 * M: number of rows
 * d: number of columns
 */
__global__ void phi_qk_ksum_kernel(const float* Q, const float* K, float* phi_q, float* phi_k_T, float* ksum, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / d;
    int col = idx % d;
    if (idx < M * d) {
        phi_q[idx] = phi(Q[idx]);
        phi_k_T[col * M + row] = phi(K[idx]);
        atomicAdd(&ksum[col], phi(K[idx]));
    }
}


/* Compute output = phi(Q) @ sum(phi(K)) */
__global__ void mv_kernel(const float * phi_q, const float* ksum, float* output, int M, int d) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_id = blockIdx.x * 4 + warp_id;
    float sum = 0.0f;

    if (row_id < M) { 
        for (int col = lane_id; col < d; col += 32) {
            sum += phi_q[row_id * d + col] * ksum[col];
        }
        __syncwarp();
        sum = warp_all_reduce_sum(sum);
        if (lane_id == 0) {
            output[row_id] = sum;
        }
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

__global__ void qkv_div_qk_kernel(const float* qkv, const float* phi_qk, float* output, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / d;
    int col = idx % d;
    if (idx < M * d) {
        output[idx] = qkv[idx] / phi_qk[row];
    }
}



// Q, K, V, output are device pointers
void linear_self_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    char* d_buf;
    size_t buf_size = d * sizeof(float) + // for ksum
                    M * d * sizeof(float) + // for phi_q
                    M * d * sizeof(float) + // for phi_k_T
                    M * sizeof(float) + // for qk
                    d * d * sizeof(float) + // for kv
                    M * d * sizeof(float); // for qkv
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buf, buf_size));
    float* ksum = (float*)d_buf;
    float* phi_q = ksum + d;
    float* phi_k_T = phi_q + M * d;
    float* phi_qk = phi_k_T + M * d;
    float* kv = phi_qk + M;
    float* qkv = kv + d * d;

    // Initialize ksum to zero
    zeros_init_kernel<<<div_up(d, 128), 128>>>(ksum, d);

    // Compute phi(Q) and phi(K)^T and sum of phi(K)
    phi_qk_ksum_kernel<<<div_up(M * d, 128), 128>>>(Q, K, phi_q, phi_k_T, ksum, M, d);

    // Compute phi_qk = phi(Q) @ sum(phi(K)) ->  (M x d) @ (d x 1) = (M x 1)
    // each warp computes a row of the output
    mv_kernel<<<div_up(M, 4), 128>>>(phi_q, ksum, phi_qk, M, d);

    // // compute kv = phi(K)^T @ V -> (d x M) @ (M x d) = (d x d)
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(d, BLOCK_N), div_up(d, BLOCK_M));
    matmul_pipline_kernel<<<grid, block>>>(phi_k_T, V, kv, d, d, M);

    // // compute qkv = phi(Q) @ kv -> (M x d) @ (d x d) = (M x d)
    grid = dim3(div_up(d, BLOCK_N), div_up(M, BLOCK_M));
    matmul_pipline_kernel<<<grid, block>>>(phi_q, kv, qkv, M, d, d);

    // // compute output = qkv / phi_qk -> (M x d) / (M x 1)
    qkv_div_qk_kernel<<<div_up(M * d, 128), 128>>>(qkv, phi_qk, output, M, d);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


torch::Tensor linear_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
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

    int M = Q.size(0);
    int d_model = Q.size(1);
    torch::Tensor output = torch::zeros({M, d_model}, Q.options());

    linear_self_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, d_model
    );
    return output;
}