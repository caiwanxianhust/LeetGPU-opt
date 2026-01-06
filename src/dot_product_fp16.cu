#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda.h>


/*
Implement a GPU program that computes the dot product of two vectors containing 16-bit floating point numbers (FP16/half). The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors 
 and 
 of length 
 is defined as:

All inputs are stored as 16-bit floating point numbers (FP16/half). For best precision, accumulation during multiplication should use FP32 before converting the final result to FP16.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
The final result must be stored in the output variable as half
Example 1:
Input:  A = [1.0, 2.0, 3.0, 4.0]
               B = [5.0, 6.0, 7.0, 8.0]
       Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)
Example 2:
Input:  A = [0.5, 1.5, 2.5]
               B = [2.0, 3.0, 4.0]
       Output: result = 15.5  (0.5*2.0 + 1.5*3.0 + 2.5*4.0)
Constraints
A and B have identical lengths
1 ≤ N ≤ 100,000,000
*/

static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void dot_product_fp16_in_block_kernel(const half* A, const half* B, half* output, int N, int nums_per_block) {
    float sum = 0.0f;
    const int offset = blockIdx.x * nums_per_block;
    for (int i = threadIdx.x; i < nums_per_block && offset + i < N; i += blockDim.x) {
        sum += __half2float(A[offset + i]) * __half2float(B[offset + i]);
    }
    __syncthreads();

    sum = block_all_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = __float2half(sum);
    }
}


__global__ void reduce_sum_fp16_kernel(const half* input, half* output, int N) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += __half2float(input[i]);
    }
    __syncthreads();

    sum = block_all_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[0] = __float2half(sum);
    }
}


// A, B, result are device pointers
void dot_product_fp16_cuda(const half* A, const half* B, half* result, int N) {
    constexpr int nums_per_block = 1024 * 32;
    int num_blocks = div_up(N, nums_per_block);
    if (num_blocks == 1) {
        dot_product_fp16_in_block_kernel<<<1, 1024>>>(A, B, result, N, nums_per_block);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        return;
    }
    half* block_sums;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&block_sums, num_blocks * sizeof(half)));
    dot_product_fp16_in_block_kernel<<<num_blocks, 1024>>>(A, B, block_sums, N, nums_per_block);
    reduce_sum_fp16_kernel<<<1, 1024>>>(block_sums, result, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(block_sums));
}