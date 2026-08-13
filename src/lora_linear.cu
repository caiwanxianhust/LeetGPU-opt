#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/* 
https://leetgpu.com/challenges/lora-linear
Implement a LoRA (Low-Rank Adaptation) linear layer forward pass. Given an input matrix x of shape batch × d_in, a base weight matrix W of shape d_out × d_in, a LoRA down-projection matrix A of shape rank × d_in, and a LoRA up-projection matrix B of shape d_out × rank, compute output = x × WT + lora_scale × (x × AT) × BT. All tensors are float32.

Implementation Requirements
Implement the solve function; do not change its signature.
Do not use external libraries beyond those provided.
Write the result into output.

Constraints
1 ≤ batch ≤ 1,024
1 ≤ d_in, d_out ≤ 8,192
1 ≤ rank ≤ 256; rank < min(d_in, d_out)
All tensors are float32 on GPU.
Performance is measured with batch = 256, d_in = 4,096, d_out = 4,096, rank = 64
*/


namespace {

// 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
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
    int lanex = laneId & 7; // 0-7
    int laney = laneId >> 3; // 0-3
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

                // 从共享内存加载 B 矩阵到寄存器，每次加载连续的 4 个 float
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2; // 计算 B 矩阵中对应的列索引
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c ^ (tk << 2)]);
                }
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
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
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
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_kernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

static __global__ void vector_add_scale_kernel(const float* A, const float* B, float* output, int size, float scale) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        output[i] = A[i] + scale * B[i];
    }
}


static void launch_vector_add_scale_kernel(const float* A, const float* B, float* output, int size, float scale) 
{
    int block_size = 256;
    int num_blocks;
    if (size < 1024 * block_size) {
        num_blocks = div_up(size, block_size);
    } else {
        num_blocks = 1024; // 最大线程块数量限制
    }
    
    vector_add_scale_kernel<<<num_blocks, block_size>>>(A, B, output, size, scale);
}


/**
 * LoRA 线性层前向计算的 CUDA 实现
 * 输入：
 * - x: 输入矩阵，形状为 (batch, d_in)
 * - W: 基础权重矩阵，形状为 (d_out, d_in)
 * - A: LoRA 下投影矩阵，形状为 (rank, d_in)
 * - B: LoRA 上投影矩阵，形状为 (d_out, rank)
 * 输出：
 * - output: 输出矩阵，形状为 (batch, d_out)
 */
// x, W, A, B, output are device pointers
void lora_linear_cuda(const float* x, const float* W, const float* A, const float* B, float* output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) 
{
    float* d_buf;
    size_t buf_size = batch * d_out * sizeof(float) + 
                      batch * rank * sizeof(float) + 
                      batch * d_out * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));

    float* xw = d_buf;
    float* xa = xw + batch * d_out;
    float* xab = xa + batch * rank;
    // xw = x * WT
    launch_matmul_rc_kernel(x, W, xw, batch, d_out, d_in);
    // xa = x * AT
    launch_matmul_rc_kernel(x, A, xa, batch, rank, d_in);
    // xab = xa * BT
    launch_matmul_rc_kernel(xa, B, xab, batch, d_out, rank);
    // // 将 xw 和 lora_scale * xab 相加得到最终输出
    launch_vector_add_scale_kernel(xw, xab, output, batch * d_out, lora_scale);
    
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


torch::Tensor lora_linear(
    const torch::Tensor& x,
    const torch::Tensor& W,
    const torch::Tensor& A,
    const torch::Tensor& B,
    float lora_scale
) { 
    int batch = x.size(0);
    int d_in = x.size(1);
    int d_out = W.size(0);
    int rank = A.size(0);
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(W.ndimension() == 2, "W must be 2D");
    TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "B must be 2D");
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(W.device().is_cuda(), "W must be on CUDA"); 
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be of type float32");
    TORCH_CHECK(W.dtype() == torch::kFloat32, "W must be of type float32");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be of type float32");
    TORCH_CHECK(x.size(1) == W.size(1), "The number of columns in x must match the number of columns in W");
    TORCH_CHECK(x.size(1) == A.size(1), "The number of columns in x must match the number of columns in A");
    TORCH_CHECK(W.size(0) == B.size(0), "The number of rows in W must match the number of rows in B");
    TORCH_CHECK(A.size(0) == B.size(1), "The number of rows in A must match the number of columns in B");

    torch::Tensor output = torch::empty({batch, d_out}, x.options());
    lora_linear_cuda(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, d_in, d_out, rank, lora_scale
    );
    return output;
}