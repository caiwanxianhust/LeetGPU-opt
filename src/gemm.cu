#include "utils.cuh"

#include <cuda_fp16.h>

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
__global__ void gemm_kernel(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
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


// A, B, and C are device pointers
void gemm_cuda(const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta) {
    // 单个 block 处理 32 * 32 * 16 的矩阵乘法
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    dim3 block(BM * BN);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_kernel<BM, BN, BK><<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
