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
        for (int i=0; i<K; ++i) {
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
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
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
    matmul_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matrix_multiplication_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    // matmul_naive_fp32(A, B, C, M, N, K);
    matmul_warp_fp32(A, B, C, M, N, K);
}
