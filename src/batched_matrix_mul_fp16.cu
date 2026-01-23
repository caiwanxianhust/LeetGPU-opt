#include "utils.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>


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


void batched_matrix_mul_naive_fp16(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(M, TILE_SIZE), BATCH);
    batched_matrix_mul_fp16_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


template <int BM, int BN, int BK, int WM, int WN>
__global__ void batched_matrix_mul_warp_wmma_fp16_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) { 
    using namespace nvcuda;

    const int warp_id = threadIdx.x / 32;
    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    __shared__ half s_a[BM * BK];
    __shared__ half s_b[BK * BN];
    __shared__ float s_c[BM * BN];

    half* a_ptr = s_a + (warp_row_id * WM) * BK;
    half* b_ptr = s_b + warp_col_id * WN;
    float* c_ptr = s_c + (warp_row_id * WM) * BN + warp_col_id * WN;

    using FragType_A = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragType_B = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FragType_C = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragType_A frag_a;
    FragType_B frag_b;
    FragType_C frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += BK) { 
        // load A and B from global to shared
        for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[i] = (row_a < M && col_a < K) ? A[blockIdx.z * M * K + row_a * K + col_a] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[i] = (row_b < K && col_b < N) ? B[blockIdx.z * K * N + row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();
        
        // compute wmma
        wmma::load_matrix_sync(frag_a, a_ptr, BK);
        wmma::load_matrix_sync(frag_b, b_ptr, BN);

        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        __syncthreads();
    }

    // write back to C
    wmma::store_matrix_sync(c_ptr, frag_acc, BN, wmma::mem_row_major);
    __syncthreads();
    
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row_c = blockIdx.y * BM + i / BN;
        int col_c = blockIdx.x * BN + i % BN;
        if (row_c < M && col_c < N) {
            C[blockIdx.z * M * N + row_c * N + col_c] = __float2half(s_c[i]);
        }
    }
}

void batched_matrix_mul_warp_wmma_fp16(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int WM = 16;
    constexpr int WN = 16;

    dim3 block(BM * BN / (WN * WM) * 32);
    dim3 grid(div_up(N, BN), div_up(M, BM), BATCH);
    batched_matrix_mul_warp_wmma_fp16_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// A, B, C are device pointers
void batched_matrix_mul_fp16_cuda(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    // batched_matrix_mul_naive_fp16(A, B, C, BATCH, M, N, K);
    batched_matrix_mul_warp_wmma_fp16(A, B, C, BATCH, M, N, K);
}