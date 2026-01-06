#include "utils.cuh"

#include <cuda.h>
#include <cuda_fp16.h>


/*
Implement a batched matrix multiplication in FP16. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
All matrices are stored in row-major order and use 16-bit floating point numbers (FP16/half). Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
The final result must be stored in the C array as half
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


namespace { 
    constexpr int TILE_SIZE = 32;
    __host__ __device__ __forceinline__ int div_up(int x, int y) {
        return (x + y - 1) / y;
    }
}


__global__ void batched_matrix_mul_fp16_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    __shared__ half s_a[TILE_SIZE * TILE_SIZE];
    __shared__ half s_b[TILE_SIZE * TILE_SIZE];

    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;
    
    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // load A and B tiles into shared memory
        for (int  i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_a = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col_a = k + i % TILE_SIZE;
            s_a[i] = (row_a < M && col_a < K) ? A[blockIdx.z * M * K + row_a * K + col_a] : __float2half(0.0f);
        }
        for (int  i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_b = k + i / TILE_SIZE;
            int col_b = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_b[i] = (row_b < K && col_b < N) ? B[blockIdx.z * K * N + row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();

        if (row_c < M && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += __half2float(s_a[ty * TILE_SIZE + i]) * __half2float(s_b[i * TILE_SIZE + tx]);
            }
        }

        // compute partial product
        __syncthreads();
    }
    // write back to C
    if (row_c < M && col_c < N) {
        C[blockIdx.z * M * N + row_c * N + col_c] = __float2half(acc);
    }
}



// A, B, C are device pointers
void batched_matrix_mul_fp16_cuda(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(M, TILE_SIZE), BATCH);
    batched_matrix_mul_fp16_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}