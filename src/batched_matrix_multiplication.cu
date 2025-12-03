#include "utils.cuh"

#include <cuda.h>

/*
Implement a batched matrix multiplication in FP32. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
All matrices are stored in row-major order and use 32-bit floating point numbers (FP32).

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the C array
Example 1:
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
Constraints
1 ≤ B ≤ 128
1 ≤ M, N, K ≤ 1024
*/

static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


template <int BM, int BN, int BK>
__global__ void batched_matmul_kernel(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) { 
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    const float *a_ptr = A + blockIdx.z * M * K;
    const float *b_ptr = B + blockIdx.z * K * N;
    float *c_ptr = C + blockIdx.z * M * N;

    int tx = threadIdx.x % (BN);
    int ty = threadIdx.x / (BN);
    int row_c = blockIdx.y * BM + ty;
    int col_c = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int i = 0; i < K; i += BK) {
        // load a from global to shared
        for (int j = threadIdx.x; j < BM * BK; j += blockDim.x) {
            int row_a = j / BK;
            int col_a = j % BK;
            if (row_a + blockIdx.y * BM < M && col_a + i < K) {
                sA[j] = a_ptr[(blockIdx.y * BM + row_a) * K + (i + col_a)];
            } else {
                sA[j] = 0.0f;
            }
        }

        for (int j = threadIdx.x; j < BK * BN; j += blockDim.x) {
            int row_b = j / BN;
            int col_b = j % BN;
            if (row_b + i < K && col_b + blockIdx.x * BN < N) {
                sB[j] = b_ptr[(i + row_b) * N + (blockIdx.x * BN + col_b)];
            } else {
                sB[j] = 0.0f;
            }
        }
        __syncthreads();

        // compute
        if (row_c < M && col_c < N) {
            for (int k = 0; k < BK; ++k) {
                acc += sA[ty * BK + k] * sB[k * BN + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        c_ptr[row_c * N + col_c] = acc;
    }
}


void batched_matmul_cuda(const float* A, const float* B, float* C, int BATCH, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 threads(BN * BM);
    dim3 blocks(div_up(N, BN), div_up(M, BM), BATCH);
    batched_matmul_kernel<BM, BN, BK><<<blocks, threads>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
} 