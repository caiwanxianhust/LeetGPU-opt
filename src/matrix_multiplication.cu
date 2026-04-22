#include "utils.cuh"

#include <assert.h>


/*
Write a program that multiplies two matrices of 32-bit floating point numbers on a GPU. Given matrix
 of dimensions
 and matrix
 of dimensions
, compute the product matrix
, which will have dimensions
. All matrices are stored in row-major format.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in matrix C

Constraints
1 ≤ M, N, K ≤ 8192
Performance is measured with M = 8192, N = 6144, K = 4096
*/


static __device__ __host__ __forceinline__ int div_up(int x, int y) {
    return (x + y - 1) / y;
}


__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float vsum = 0.0f;
    if (row < M && col < N) {
        for (int i = 0; i < K; ++i) {
            vsum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = vsum;
    }
}


/**
 * 矩阵乘法
 * @param C 输出矩阵，行主序
 * @param A 输入矩阵，行主序
 * @param B 输入矩阵，行主序
 * @param lda 输入矩阵的主维度
 * @param ldb 输入矩阵的主维度
 * @param ldc 输出矩阵的主维度
 */
__device__ void mma_m16n16k16_fp32(float* C, const float* A, const float* B, int lda, int ldb, int ldc) {
    const int lane_id = threadIdx.x % 32;
    int row_c = lane_id / 8;
    int col_c = lane_id % 8;
    for (int row = row_c; row < 16; row += 4) {
        for (int col = col_c; col < 16; col += 8) {
            float acc = 0.0f;
            float a_val, b_val;
#pragma unroll
            for (int k = 0; k < 16; k++) {
                a_val = (A[row * lda + k]);
                b_val = (B[k * ldb + col]);
                acc += a_val * b_val;
            }
            C[row * ldc + col] = acc + C[row * ldc + col];
            __syncwarp();
        }
    }
}


template < int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
__global__ void matmul_warp_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK;
    const int ldb = BN;
    const int ldc = BN;
    __shared__ float s_a[BM * lda];
    __shared__ float s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    // warp 排布 2x2
    constexpr int warp_col = 2;
    constexpr int warp_row = 2;
    // assert(warp_col * warp_row == blockDim.x / 32);

    // warp 在 BM 和 BN 轴上分别循环 2 次
    constexpr int M_LOOP = BM / (warp_row * MMA_M);
    constexpr int N_LOOP = BN / (warp_col * MMA_N);
    // warp 在 BK 轴上循环 2 次
    constexpr int K_LOOP = BK / (MMA_K);

    // 当前 warp 的索引
    const int warp_col_id = warp_id % (warp_col);
    const int warp_row_id = warp_id / (warp_col);

    // initialize C in shared by zero
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : (0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : (0.0f);
        }
        __syncthreads();

        // compute block mma 64 x 64
        for (int m_id = 0; m_id < M_LOOP; m_id++) {
            for (int n_id = 0; n_id < N_LOOP; n_id++) {
                int a_off = m_id * (MMA_M * warp_row) * lda + warp_row_id * MMA_M * lda;
                int b_off = n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                int c_off = m_id * (MMA_M * warp_row) * ldc + warp_row_id * MMA_M * ldc + n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                for (int k_id = 0; k_id < K_LOOP; k_id++) {
                    a_off += k_id * MMA_K;
                    b_off += k_id * MMA_K * ldb;
                    mma_m16n16k16_fp32(s_c + c_off, s_a + a_off, s_b + b_off, lda, ldb, ldc);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = s_c[row * ldc + col];
        }
    }
}


void matmul_naive_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel << <blocksPerGrid, threadsPerBlock >> > (A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void matmul_warp_fp32(const float* A, const float* B, float* C, int M, int N, int K) {
    // 每个 warp 计算一个 16x16 的子矩阵，总共 4 个 warp 计算一个 32x32 的子矩阵
    // 每个 block 计算一个 64 x 64 的子矩阵，所以 4 个 warp 分别在 M 和 N 方向上循环 2 次
    // BK = 32, 每个 warp 计算 16x16 矩阵时，需要在 K 维度上计算两次 mma 并累加
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int BK = 32;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int block_size = 128; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K> << <grid, block >> > (A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}






/**
 * 加载函数：Global Memory -> Shared Memory
 * 保持之前的修复：安全读取 float，组装 float4
 */
template <int BM, int BN, int BK>
__device__ void loadFromGmem(const float *A, const float *B, const int M, const int N, const int K,
                             float *s_a, float *s_b, int offset_a, int offset_b,
                             const int bkidx, const int buffer_id, const int extra_col)
{
    int row_a, col_a, row_b, col_b;
    
    // --- 加载 A 矩阵 ---
    #pragma unroll
    for (int j = (threadIdx.x << 2); j < BM * BK; j += (blockDim.x << 2))
    {
        row_a = j / BK;
        col_a = j % BK;

        int gmem_row = offset_a / K + row_a;
        int gmem_col = bkidx + col_a;

        float4 tmp;
        if (gmem_row < M && gmem_col < K) {
            const float* src_ptr = A + offset_a + row_a * K + bkidx + col_a;
            tmp.x = src_ptr[0];
            tmp.y = (col_a + 1 < BK && gmem_col + 1 < K) ? src_ptr[1] : 0.0f;
            tmp.z = (col_a + 2 < BK && gmem_col + 2 < K) ? src_ptr[2] : 0.0f;
            tmp.w = (col_a + 3 < BK && gmem_col + 3 < K) ? src_ptr[3] : 0.0f;
        } else {
            tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        int base_idx = buffer_id * BK * (BM + extra_col) + row_a;
        s_a[base_idx + col_a * (BM + extra_col)] = tmp.x;
        s_a[base_idx + (col_a + 1) * (BM + extra_col)] = tmp.y;
        s_a[base_idx + (col_a + 2) * (BM + extra_col)] = tmp.z;
        s_a[base_idx + (col_a + 3) * (BM + extra_col)] = tmp.w;
    }

    // --- 加载 B 矩阵 ---
    #pragma unroll
    for (int j = (threadIdx.x << 2); j < BK * BN; j += (blockDim.x << 2))
    {
        row_b = j / BN;
        col_b = j % BN;

        int gmem_row = bkidx + row_b;
        int gmem_col = offset_b + col_b;

        float4 val;
        if (gmem_row < K && gmem_col < N) {
            const float* src_ptr = B + gmem_row * N + gmem_col;
            val.x = src_ptr[0];
            val.y = (col_b + 1 < BN && gmem_col + 1 < N) ? src_ptr[1] : 0.0f;
            val.z = (col_b + 2 < BN && gmem_col + 2 < N) ? src_ptr[2] : 0.0f;
            val.w = (col_b + 3 < BN && gmem_col + 3 < N) ? src_ptr[3] : 0.0f;
        } else {
            val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        reinterpret_cast<float4 *>(s_b + buffer_id * BK * BN + row_b * BN + col_b)[0] = val;
    }
}

/**
 * 计算函数：Shared Memory -> Registers -> Accumulators
 */
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__device__ void processFromSmem(const float *s_a, const float *s_b, float *reg_a, float *reg_b, float *tmp, const int WMITERS, const int WNITERS,
                                const int warp_sub_M, const int warp_sub_N, const int warp_row, const int warp_col, const int thread_in_warp_row,
                                const int thread_in_warp_col, const int buffer_id, const int extra_col)
{
    #pragma unroll
    for (int j = 0; j < BK; ++j)
    {
        // Load A
        #pragma unroll
        for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
        {
            #pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                reg_a[w_m_id * TM + tmidx] =
                    s_a[buffer_id * BK * (BM + extra_col) + j * (BM + extra_col) + warp_row * WM + w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx];
            }
        }

        // Load B
        #pragma unroll
        for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
        {
            #pragma unroll
            for (int tnidx = 0; tnidx < TN; ++tnidx)
            {
                reg_b[w_n_id * TN + tnidx] =
                    s_b[buffer_id * BK * BN + j * BN + warp_col * WN + w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx];
            }
        }

        // MAC
        #pragma unroll
        for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
        {
            #pragma unroll
            for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
            {
                #pragma unroll
                for (int tmidx = 0; tmidx < TM; ++tmidx)
                {
                    #pragma unroll
                    for (int tnidx = 0; tnidx < TN; ++tnidx)
                    {
                        tmp[w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx] +=
                            reg_a[w_m_id * TM + tmidx] * reg_b[w_n_id * TN + tnidx];
                    }
                }
            }
        }
    }
}

/**
 * 写回函数：Registers -> Global Memory
 * 修复点：完全放弃向量化写入，改用安全的标量写入，彻底杜绝越界。
 */
template <int BM, int BN, int BK, int TM, int TN>
__device__ void writeFromRegToGmem(const float *tmp, float *C, const int M, const int N,
                                   const int WMITERS, const int WNITERS,
                                   const int warp_sub_M, const int warp_sub_N,
                                   const int thread_in_warp_row, const int thread_in_warp_col,
                                   const int block_start_m, const int block_start_n,
                                   const int warp_row, const int warp_col,
                                   const float alpha, const float beta)
{
    // 计算当前 Warp 负责的区域起点
    int warp_start_m = block_start_m + warp_row * (WMITERS * warp_sub_M);
    int warp_start_n = block_start_n + warp_col * (WNITERS * warp_sub_N);

    #pragma unroll
    for (int w_m_id = 0; w_m_id < WMITERS; ++w_m_id)
    {
        #pragma unroll
        for (int w_n_id = 0; w_n_id < WNITERS; ++w_n_id)
        {
            #pragma unroll
            for (int tmidx = 0; tmidx < TM; ++tmidx)
            {
                #pragma unroll
                // 这里不再按 4 步进，而是按 1 步进，确保每个元素都能被单独检查
                for (int tnidx = 0; tnidx < TN; ++tnidx)
                {
                    // 计算全局坐标
                    int row_c = warp_start_m + w_m_id * warp_sub_M + thread_in_warp_row * TM + tmidx;
                    int col_c = warp_start_n + w_n_id * warp_sub_N + thread_in_warp_col * TN + tnidx;
                    
                    // 计算 tmp 数组中的索引
                    int tmp_id = w_m_id * WNITERS * TM * TN + w_n_id * TM * TN + tmidx * TN + tnidx;

                    // --- 严格边界检查 ---
                    if (row_c < M && col_c < N) {
                        // 读取旧值 (如果 beta != 0)
                        float c_val = C[row_c * N + col_c];
                        // 写入新值
                        C[row_c * N + col_c] = alpha * tmp[tmp_id] + beta * c_val;
                    }
                }
            }
        }
    }
}

/**
 * 主 Kernel
 */
template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void SgemmSmemKernel_double_buffer(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                              const float alpha, const float beta)
{
    int block_start_m = blockIdx.y * BM;
    int block_start_n = blockIdx.x * BN;

    const int warp_row = (threadIdx.x >> 5) / (BN / WN);
    const int warp_col = (threadIdx.x >> 5) % (BN / WN);

    const int WMITERS = 2;
    const int WNITERS = (WM * WN) / (TM * TN * 32 * WMITERS);
    const int warp_sub_M = WM / WMITERS;
    const int warp_sub_N = WN / WNITERS;

    const int thread_in_warp_row = (threadIdx.x & 0x1f) / (warp_sub_N / TN);
    const int thread_in_warp_col = (threadIdx.x & 0x1f) % (warp_sub_N / TN);

    const int offset_a = block_start_m * K;
    const int offset_b = block_start_n;

    const int extra_col = 4;
    __shared__ float s_a[2 * BK * (BM + extra_col)];
    __shared__ float s_b[2 * BK * BN];

    float tmp[WMITERS * TM * WNITERS * TN] = {0.0f};
    float reg_a[WMITERS * TM];
    float reg_b[WNITERS * TN];

    int buffer_id = 0;

    if (0 < K) {
        loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, 0, buffer_id, extra_col);
    }
    __syncthreads();

    for (int k_offset = 0; k_offset < K; k_offset += BK)
    {
        int next_k_offset = k_offset + BK;
        int next_buffer_id = 1 - buffer_id;

        if (next_k_offset < K) {
            loadFromGmem<BM, BN, BK>(A, B, M, N, K, s_a, s_b, offset_a, offset_b, next_k_offset, next_buffer_id, extra_col);
        }

        processFromSmem<BM, BN, BK, WM, WN, TM, TN>(s_a, s_b, reg_a, reg_b, tmp, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                                    warp_row, warp_col, thread_in_warp_row, thread_in_warp_col, buffer_id, extra_col);

        buffer_id = next_buffer_id;
        __syncthreads();
    }

    writeFromRegToGmem<BM, BN, BK, TM, TN>(tmp, C, M, N, WMITERS, WNITERS, warp_sub_M, warp_sub_N,
                                           thread_in_warp_row, thread_in_warp_col, block_start_m, block_start_n,
                                           warp_row, warp_col, alpha, beta);
}

/**
 * 启动函数
 */
void launchSgemmSmemKernel_double_buffer(const float *A, const float *B, float *C, const int M, const int N, const int K,
                                         const float alpha, const float beta, cudaStream_t stream)
{
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int WM = 64;
    const int WN = 32;
    const int TM = 4;
    const int TN = 4;

    dim3 block(BN * BM / (WM * WN) * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    SgemmSmemKernel_double_buffer<BM, BN, BK, WM, WN, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
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



__global__ void matmul_advanced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1. 计算当前 Block 负责的全局坐标起点
    int col0 = blockIdx.x * BLOCK_N; // 当前 Block 处理 N 维度的起始列
    int row0 = blockIdx.y * BLOCK_M; // 当前 Block 处理 M 维度的起始行
    int tidx = threadIdx.x;          // 当前线程在 Block 内的 ID (0-255)

    // 2. 共享内存声明 (使用双缓冲 [2][...])
    // sA: [2][8][128] - 存储 A 矩阵分块，转置存储以优化访问
    // sB: [2][8][128] - 存储 B 矩阵分块
    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];

    // 3. 线程 ID 映射计算 (核心魔法开始)
    
    // --- A 矩阵加载线程映射 ---
    // tAx: 线程在 8x32 网格中的列 ID (0-7)
    int tAx = tidx & (THREAD_A_LAYOUT_X - 1); 
    // tAy: 线程在 8x32 网格中的行 ID (0-31)
    int tAy = tidx / THREAD_A_LAYOUT_X;

    // --- B 矩阵加载线程映射 ---
    // tBx: 线程在 32x8 网格中的列 ID (0-31)
    int tBx = tidx & (THREAD_B_LAYOUT_X - 1);
    // tBy: 线程在 32x8 网格中的行 ID (0-7)
    int tBy = tidx / THREAD_B_LAYOUT_X;
    
    // --- C 矩阵计算线程映射 (Warp 级重组) ---
    int warpId = tidx >> 5;        // 当前 Warp ID (0-7)
    int laneId = tidx & 31;        // Warp 内的线程 ID (0-31)
    
    // 将 Warp 重新组织为 2D 网格
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // Warp 在 X 方向的 ID
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // Warp 在 Y 方向的 ID
    
    // 将 Lane 重新组织为 2D 网格 (这里使用了特殊的位运算映射，为了配合 XOR Bank Conflict 消除)
    int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3
    
    // 计算当前线程在 C 矩阵计算网格中的最终坐标
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // 4. 寄存器声明
    // acc: 每个线程维护一个 8x8 的累加器矩阵，用于存储计算结果
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    // tCsA, tCsB: 寄存器缓存，用于存储从 Shared Memory 加载的数据
    // [2][...] 用于双缓冲流水线
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];
    
    int bufferId = 0; // 双缓冲索引 (0 或 1)

    // ==========================================
    // 5. 预加载阶段 (加载 K=0 的数据块)
    // ==========================================
    
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy; // 全局行号
        // 关键 XOR 操作：(i + tAy) ^ ((tAx) << 2)
        // 这行代码通过异或运算打乱共享内存地址，彻底消除 Bank Conflict
        sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f; 
    }
    
    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = i + tBy; // 全局行号 (K 维度)
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx; // 全局列号 (N 维度)
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f; 
        }
    }
    __syncthreads(); // 等待所有线程加载完毕

    // ==========================================
    // 6. 主循环：细粒度流水线
    // ==========================================
    // 循环范围是 BLOCK_K 到 K + BLOCK_K
    // 这种写法是为了在最后一次迭代中只计算不加载
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        
        // --- 内部流水线循环 ---
        // 这里的 tk 从 0 到 BLOCK_K (包含)，共 9 次迭代
        // 目的是在一个循环内完成：计算(tk-1) + 加载(tk)
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++ tk) {
            
            // A. 计算阶段 (使用上一轮加载的数据)
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm ++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn ++) {
                        // 累加：acc += A_reg * B_reg
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }
            
            // B. 加载阶段 (从 Shared Memory 搬运到寄存器)
            if (tk < BLOCK_K) {
                // 加载 A 到寄存器 tCsA
                // 这里使用 float4 向量化加载，一次搬运 4 个 float
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm ++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2; // 计算行偏移
                    // 注意这里也有 XOR 地址映射，与加载到 Shared Memory 时对应
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
                // 加载 B 到寄存器 tCsB
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2; // 计算列偏移
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }
        
        // ==========================================
        // 7. 预加载下一块 K 数据 (Global -> Shared)
        // ==========================================
        if (k < K) {
            int c = k + tAx; // 下一块 K 的列坐标
            // 加载 A 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f; 
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
            __syncthreads(); // 同步，确保下一块数据准备好
        }
        bufferId ^= 1; // 切换双缓冲索引 (0 -> 1, 1 -> 0)
    } 

    // ==========================================
    // 8. 写回结果 (Register -> Global Memory)
    // ==========================================
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i ++) {
        // 复杂的位运算重组坐标，确保写回时也是合并访问
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j ++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void launch_matmul_advanced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    printf("Launching matmul_advanced_kernel with %d x %d x %d\n", M, N, K);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);
    
    matmul_advanced_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
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
        sA[0][tAx][(j + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
    }
    

    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_Y) {
        int r = i + tBy; // K 维度的列索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx; // N 维度的列  
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
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
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
                sA[bufferId ^ 1][tAx][(j + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
            

            // 加载 B 矩阵的下一块数据
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+= THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy; // K 维度的列索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+= THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx; // N 维度的列  
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
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3); // C 的列索引，配合寄存器加载的列偏移，对应于一次加载 float4
            // int c = col0 + tCx * THREAD_C_X_TILE_SIZE + j; // C 的列索引，配合寄存器加载的列偏移，对应于一次加载 float
            // int c = col0 + (tCx << 1) + (j & (~1)) * THREAD_C_LAYOUT_X + (j & 1); // C 的列索引，配合寄存器加载的列偏移, 对应于一次加载 float2
            // int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7); // C 的列索引，配合寄存器加载的列偏移, 对应于一次加载 float8
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


__global__ void matmul_rc_adavanced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    int row0 = blockIdx.y * BLOCK_M;
    int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明 (使用双缓冲 [2][...])
    // sA: [2][8][128] - 存储 A 矩阵分块，转置存储以优化访问
    // sB: [2][8][128] - 存储 B 矩阵分块，转置存储以优化访问
    __shared__ float sA[2][BLOCK_K][BLOCK_M];
    __shared__ float sB[2][BLOCK_K][BLOCK_N];

    // 3. 线程 id 映射计算 (核心魔法开始)
    // --- A 矩阵加载线程映射 ---
    int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 线程在 32x8 网格中的列 id, 0-7
    int tAy = threadIdx.x / THREAD_A_LAYOUT_X;       // 线程在 32x8 网格中的行 id, 0-31

    // --- B 矩阵加载线程映射 ---
    int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 线程在 8x32 网格中的列 id, 0-31
    int tBy = threadIdx.x / THREAD_B_LAYOUT_X;       // 线程在 8x32 网格中的行 id, 0-7

    // --- C 矩阵计算线程映射 (Warp 级重组) ---
    int warpId = threadIdx.x >> 5;        // 当前 warp id, 0-7
    int laneId = threadIdx.x & 31;        // warp 内的线程 id, 0-31

    // 将 warp 重新组织为 2D 网格(4x2)
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // warp 在 x 方向的 id, 0-1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // warp 在 y 方向的 id, 0-3

    // 将 lane 重新组织为 2D 网格(4x8) (这里使用了特殊的位运算映射，为了配合 XOR Bank Conflict 消除)
    int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3

    // 计算当前线程在 C 矩阵计算网格中的最终坐标
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15

    // 4. 寄存器声明
    // acc: 每个线程维护一个 8x8 的累加器矩阵，用于存储计算结果
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    // tCsA, tCsB: 寄存器缓存，用于存储从 Shared Memory 加载的数据
    // [2][...] 用于双缓冲流水线
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];

    int bufferId = 0; // 双缓冲索引 (0 或 1)

    // 5. 预加载阶段 (加载 K=0 的数据块)
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy; // 全局行号
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
            int c = j + tAx;    // 全局列号
            // 关键 XOR 操作：(i + tAy) ^ ((tAx) << 2)
            // 这行代码通过异或运算打乱共享内存地址，彻底消除 Bank Conflict
            sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
        
    }
    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_LAYOUT_X) {
        int r = col0 + i + tBx; // 全局行号 (N 维度) - 注意这里 r 变成了 N 的索引
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_LAYOUT_Y) {
            int c = j + tBy;    // 全局列号 (K 维度) - 注意这里 c 变成了 K 的索引
            
            // 写入 Shared: sB[K][N] -> sB[c][r] (注意下标交换，实现转置)
            // 这里的 c 对应 sB 的行(K)，r 对应 sB 的列(N)
            sB[0][j + tBy][(i + tBx) ^ ((j + tBy) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环：细粒度流水线
    // 循环范围是 BLOCK_K 到 K + BLOCK_K
    // 这种写法是为了在最后一次迭代中只计算不加载
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // --- 内部流水线循环 ---
        // 这里的 tk 从 0 到 BLOCK_K (包含)，共 9 次迭代
        // 目的是在一个循环内完成：计算(tk-1) + 加载(tk)
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; ++tk) {
            // A. 计算阶段 (使用上一轮加载的数据)
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        // 累加：acc += A_reg * B_reg
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }

            // B. 加载阶段 (使用当前 k 值加载数据)
            if (tk < BLOCK_K) { 
                // 加载 A 到寄存器 tCsA
                // 这里使用 float4 向量化加载，一次搬运 4 个 float
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm++) {
                    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2; // 计算行偏移
                    // 注意这里也有 XOR 地址映射，与加载到 Shared Memory 时对应
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
                // 加载 B 到寄存器 tCsB
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn++) {
                    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2; // 计算列偏移
                    // 还原物理地址
                    // 写入时使用了：(i + tBx) ^ (tk << 2)
                    // 读取时逆运算：逻辑地址 ^ (tk << 2)
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c ^ (tk << 2)]);
                }
            }
        }

        // 7. 预加载下一块 K 数据 (Global -> Shared)
        if (k < K) {
            // 加载 A 矩阵的下一块 (BLOCK_K x BLOCK_M)
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy; // 全局行号
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                    int c = k + j + tAx; // 全局列号
                    sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 矩阵的下一块 (BLOCK_K x BLOCK_N)
            for (int i = 0; i < BLOCK_N; i += THREAD_B_LAYOUT_X) {
                int r = col0 + i + tBx; // 全局行号 (N 维度) - 注意这里 r 变成了 N 的索引
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_LAYOUT_Y) {
                    int c = k + j + tBy;    // 全局列号 (K 维度) - 注意这里 c 变成了 K 的索引
                    
                    // 写入 Shared: sB[K][N] -> sB[c][r] (注意下标交换，实现转置)
                    // 这里的 c 对应 sB 的行(K)，r 对应 sB 的列(N)
                    sB[bufferId ^ 1][j + tBy][(i + tBx) ^ ((j + tBy) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();

        }

        bufferId = bufferId ^ 1; // 切换双缓冲索引
    }

    // 8. 写回结果 (Register -> Global Memory)
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        // 复杂的位运算重组坐标，确保写回时也是合并访问
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N) {
                C[r * N + c] = acc[i][j];
            }
        }
    }
}


void launch_matmul_rc_adavanced_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    printf("Launching matmul_rc_adavanced_kernel with %d x %d x %d\n", M, N, K);
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N,
                       (M + BLOCK_M - 1) / BLOCK_M);
    
    matmul_rc_adavanced_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
}



/** 
 * 矩阵乘法
 * 输入：
 *   A: M x K 的矩阵，存储在 GPU 全局内存中，行主序
 *   B: K x N 的矩阵，存储在 GPU 全局内存中，行主序
 * 输出：
 *   C: M x N 的矩阵，存储在 GPU 全局内存中，初始值不确定，函数内需要正确处理（如先清零或累加），行主序
 */
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matrix_multiplication_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    // matmul_naive_fp32(A, B, C, M, N, K);
    // matmul_warp_fp32(A, B, C, M, N, K);
    // launchSgemmSmemKernel_double_buffer(A, B, C, M, N, K, 1.0f, 0.0f, 0);
    launch_matmul_advanced_kernel(A, B, C, M, N, K);
    // launch_matmul_kernel(A, B, C, M, N, K);
}


/** 
 * 矩阵乘法
 * 输入：
 *   A: M x K 的矩阵，存储在 GPU 全局内存中，行主序
 *   B: N x K 的矩阵，存储在 GPU 全局内存中，行主序
 * 输出：
 *   C: M x N 的矩阵，存储在 GPU 全局内存中，初始值不确定，函数内需要正确处理（如先清零或累加），行主序
 */
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matrix_multiplication_row_col_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    launch_matmul_rc_adavanced_kernel(A, B, C, M, N, K);
}