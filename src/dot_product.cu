#include "utils.cuh"
#include "cuda_kernels.cuh"


/**
Implement a GPU program that computes the dot product of two vectors containing 32-bit floating point numbers. The dot product is the sum of the products of the corresponding elements of two vectors.

Mathematically, the dot product of two vectors 
 and 
 of length 
 is defined as:

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
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


__global__ void dot_product_kernel(float* A, float* B, float* result, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        temp += A[i] * B[i];
    }
    __syncthreads();

    temp = block_all_reduce_sum(temp);
    if (threadIdx.x == 0) {
        atomicAdd(&result[0], temp);
    }
}


void dot_product_cuda(float* A, float* B, float* result, int N) {
    int block_size = 1024;
    int grid_size;
    if (N < block_size) {
        grid_size = 1;
    }
    else if (N < 1024 * 1024) {
        grid_size = (N + block_size - 1) / block_size;
    }
    else {
        grid_size = 1024; // Limit to 1024 blocks
    }
    dot_product_kernel<<<grid_size, block_size>>>(A, B, result, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}