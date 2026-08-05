#include "utils.cuh"

#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

/*
Implement a basic General Matrix Multiplication (GEMM). Given matrix 
 of dimensions 
, matrix 
 of dimensions 
, input/output matrix 
 of dimensions 
, and scalar multipliers 
 and 
, compute the operation:

The input matrices 
, 
, and the initial state of 
 contain 16-bit floating-point numbers (FP16/half). All matrices are stored in row-major order. The scalars 
 and 
 are 32-bit floats.

Implementation Requirements
Use only native features (external libraries other than WMMA are not permitted).
The solve function signature must remain unchanged.
Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.
The final result must be stored back into matrix C as half.

Constraints
16 ≤ M, N, K ≤ 4096
*/


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
  return (a + b - 1) / b;
}

template <int BM, int BN, int BK>
__global__ void gemm_naive_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    __shared__ half s_a[BM * BK];
    __shared__ half s_b[BK * BN];

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;
    int row_c = blockIdx.y * BM + ty;
    int col_c = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int i = 0; i < K; i += BK) {
        for (int j = threadIdx.x; j < BM * BK; j += blockDim.x) {
            int row_a = j / BK + blockIdx.y * BM;
            int col_a = j % BK + i;
            s_a[j] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
        }
        for (int j = threadIdx.x; j < BK * BN; j += blockDim.x) {
            int row_b = j / BN + i;
            int col_b = j % BN + blockIdx.x * BN;
            s_b[j] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();

        if (row_c < M && col_c < N) {
            // Perform the multiplication and accumulate
            for (int j = 0; j < BK; j++) {
                acc += __half2float(s_a[ty * BK + j]) * __half2float(s_b[j * BN + tx]);
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = __float2half(alpha * acc + beta * __half2float(C[row_c * N + col_c]));
    }
}

void gemm_naive(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // 单个 block 处理 32 * 32 * 16 的矩阵乘法
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    dim3 block(BM * BN);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_naive_kernel<BM, BN, BK><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
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
__device__ void mma_m8n8k8_fp16(float* C, const half* A, const half* B, int lda, int ldb, int ldc) {
    // const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    int row_c = lane_id / 8;
    int col_c = lane_id % 8;

    for (int row = row_c; row < 8; row += 4) {
        float acc = 0.0f;
        float a_val, b_val;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            a_val = __half2float(A[row * lda + k]);
            b_val = __half2float(B[k * ldb + col_c]);
            acc += a_val * b_val;
        }
        C[row * ldc + col_c] = acc + C[row * ldc + col_c];
        __syncwarp();
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
__device__ void mma_m16n16k16_fp16(float* C, const half* A, const half* B, int lda, int ldb, int ldc) {
    // const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    int row_c = lane_id / 8;
    int col_c = lane_id % 8;
    for (int row = row_c; row < 16; row += 4) { 
        for (int col = col_c; col < 16; col += 8) {
            float acc = 0.0f;
            float a_val, b_val;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                a_val = __half2float(A[row * lda + k]);
                b_val = __half2float(B[k * ldb + col]);
                acc += a_val * b_val;
            }
            C[row * ldc + col] = acc + C[row * ldc + col];
            __syncwarp();
        }
    }
}


template < int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_warp_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ half s_a[BM * lda];
    __shared__ half s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

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
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();

        // compute block mma 
        // lda = BK, ldb = BN, ldc = BN
        half* a_ptr = s_a + warp_row_id * WM * lda;
        half* b_ptr = s_b + warp_col_id * WN;
        float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
        mma_m16n16k16_fp16(c_ptr, a_ptr, b_ptr, lda, ldb, ldc);
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = __float2half(alpha * s_c[row * ldc + col] + beta * __half2float(C[row_c * N + col_c]));
        }
    }
}

template < int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_warp_double_buffer_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ half s_a[2 * BM * lda];
    __shared__ half s_b[2 * BK * ldb];
    __shared__ float s_c[BM * ldc];

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    // initialize C in shared by zero
    #pragma unroll
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }

    // load first block
    int buffer_id = 0;
    // load A from global to shared
    #pragma unroll
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
        int row = i / BK;
        int col = i % BK;
        int row_a = i / BK + blockIdx.y * BM;
        int col_a = i % BK;
        s_a[buffer_id * BM * lda + row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
    }
    // load B from global to shared
    #pragma unroll
    for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_b = i / BN;
        int col_b = i % BN + blockIdx.x * BN;
        s_b[buffer_id * BK * ldb + row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < K - BK; k += BK) {
        // load A from global to shared
        #pragma unroll
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + (k + BK);
            s_a[(1 - buffer_id) * BM * lda + row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
        }
        // load B from global to shared
        #pragma unroll
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + (k + BK);
            int col_b = i % BN + blockIdx.x * BN;
            s_b[(1 - buffer_id) * BK * ldb + row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
        }

        // compute block mma 
        // lda = BK, ldb = BN, ldc = BN
        half* a_ptr = s_a + buffer_id * BM * lda + warp_row_id * WM * lda;
        half* b_ptr = s_b + buffer_id * BK * ldb + warp_col_id * WN;
        float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
        mma_m16n16k16_fp16(c_ptr, a_ptr, b_ptr, lda, ldb, ldc);
        __syncthreads();
        buffer_id = 1 - buffer_id;
    }

    // compute last block mma
    half* a_ptr = s_a + buffer_id * BM * lda + warp_row_id * WM * lda;
    half* b_ptr = s_b + buffer_id * BK * ldb + warp_col_id * WN;
    float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
    mma_m16n16k16_fp16(c_ptr, a_ptr, b_ptr, lda, ldb, ldc);
    __syncthreads();

    #pragma unroll
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = __float2half(alpha * s_c[row * ldc + col] + beta * __half2float(C[row_c * N + col_c]));
        }
    }
}

template < int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_warp_wmma_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) { 
    using namespace nvcuda;
    // block 内 warp 的 id
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ half s_a[BM * lda];
    __shared__ half s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    // lda = BK, ldb = BN, ldc = BN
    half* a_ptr = s_a + warp_row_id * WM * lda;
    half* b_ptr = s_b + warp_col_id * WN;

    using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FragCType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragAType frag_a;
    FragBType frag_b;
    FragCType frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();

        // compute block mma 
        wmma::load_matrix_sync(frag_a, a_ptr, lda);
        wmma::load_matrix_sync(frag_b, b_ptr, ldb);
        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        __syncthreads();
    }

    float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
    wmma::store_matrix_sync(c_ptr, frag_acc, ldc, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = __float2half(alpha * s_c[row * ldc + col] + beta * __half2float(C[row_c * N + col_c]));
        }
    }
}


template < int BM, int BN, int BK, int WM, int WN>
__global__ void gemm_warp_wmma_double_buffer_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    using namespace nvcuda;
    // block 内 warp 的 id
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ half s_a[2 * BM * lda];
    __shared__ half s_b[2 * BK * ldb];
    __shared__ float s_c[BM * ldc];

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FragCType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragAType frag_a;
    FragBType frag_b;
    FragCType frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);
    __syncthreads();

    // load first block
    int buffer_id = 0;
    // load A from global to shared
    #pragma unroll
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
        int row = i / BK;
        int col = i % BK;
        int row_a = i / BK + blockIdx.y * BM;
        int col_a = i % BK;
        s_a[buffer_id * BM * lda + row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
    }
    // load B from global to shared
    #pragma unroll
    for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_b = i / BN;
        int col_b = i % BN + blockIdx.x * BN;
        s_b[buffer_id * BK * ldb + row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < K - BK; k += BK) {
        // load A from global to shared
        #pragma unroll
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + (k + BK);
            s_a[(1 - buffer_id) * BM * lda + row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : __float2half(0.0f);
        }
        // load B from global to shared
        #pragma unroll
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + (k + BK);
            int col_b = i % BN + blockIdx.x * BN;
            s_b[(1 - buffer_id) * BK * ldb + row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : __float2half(0.0f);
        }

        // compute block mma 
        // lda = BK, ldb = BN, ldc = BN
        half* a_ptr = s_a + buffer_id * BM * lda + warp_row_id * WM * lda;
        half* b_ptr = s_b + buffer_id * BK * ldb + warp_col_id * WN;
        // compute block mma 
        wmma::load_matrix_sync(frag_a, a_ptr, lda);
        wmma::load_matrix_sync(frag_b, b_ptr, ldb);
        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        __syncthreads();
        buffer_id = 1 - buffer_id;
    }

    // compute last block mma
    half* a_ptr = s_a + buffer_id * BM * lda + warp_row_id * WM * lda;
    half* b_ptr = s_b + buffer_id * BK * ldb + warp_col_id * WN;
    wmma::load_matrix_sync(frag_a, a_ptr, lda);
    wmma::load_matrix_sync(frag_b, b_ptr, ldb);
    wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
    __syncthreads();

    float* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
    wmma::store_matrix_sync(c_ptr, frag_acc, ldc, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = __float2half(alpha * s_c[row * ldc + col] + beta * __half2float(C[row_c * N + col_c]));
        }
    }
}


void gemm_warp(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int BK = 16;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int block_size = BM * BN / (WM * WN) * 32; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_warp_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void gemm_warp_double_buffer(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int BK = 16;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int block_size = BM * BN / (WM * WN) * 32; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_warp_double_buffer_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void gemm_warp_wmma(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int BK = 16;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int block_size = BM * BN / (WM * WN) * 32; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_warp_wmma_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void gemm_warp_wmma_double_buffer(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    constexpr int WM = 16;
    constexpr int WN = 16;
    constexpr int BK = 16;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int block_size = BM * BN / (WM * WN) * 32; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_warp_wmma_double_buffer_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



// A, B, and C are device pointers
void gemm_cuda(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // gemm_naive(A, B, C, M, N, K, alpha, beta);
    // gemm_warp(A, B, C, M, N, K, alpha, beta);
    // gemm_warp_double_buffer(A, B, C, M, N, K, alpha, beta);
    // gemm_warp_wmma(A, B, C, M, N, K, alpha, beta);
    gemm_warp_wmma_double_buffer(A, B, C, M, N, K, alpha, beta);
}


torch::Tensor gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M, int N, int K,
    float alpha, float beta
) { 
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(C.ndimension() == 1, "Input C must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(C.device().is_cuda(), "Input C must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be of type float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be of type float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "Input C must be of type float16");
    TORCH_CHECK(A.size(0) == M * K, "Input A must be a 1D matrix")
    TORCH_CHECK(B.size(0) == N * K, "Input B must be a 1D matrix")
    TORCH_CHECK(C.size(0) == M * N, "Input C must be a 1D matrix")
    
    gemm_cuda(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        alpha, beta
    );
    return C;
}