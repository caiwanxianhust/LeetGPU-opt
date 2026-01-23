#include "utils.cuh"


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


template < int BM, int BN, int BK, int WM, int WN>
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

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

     // lda = BK, ldb = BN, ldc = BN
    float* a_ptr = s_a + warp_row_id * WM * lda;
    float* b_ptr = s_b + warp_col_id * WN;
    float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;

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

        // compute block mma 
        mma_m16n16k16_fp32(c_ptr, a_ptr, b_ptr, lda, ldb, ldc);
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
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int BK = 16;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int block_size = BM * BN / (WM * WN) * 32; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_warp_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matrix_multiplication_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    // matmul_naive_fp32(A, B, C, M, N, K);
    matmul_warp_fp32(A, B, C, M, N, K);
}
