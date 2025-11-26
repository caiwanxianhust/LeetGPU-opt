#include "utils.cuh"
#include "cuda_kernels.cuh"


/*
Implement a GPU program that performs sparse matrix-vector multiplication. Given a sparse matrix 
 of dimensions 
 and a dense vector 
 of length 
, compute the product vector 
, which will have length 
. A is stored in row-major order. nnz is the number of non-zero elements in A.

Mathematically, the operation is defined as:

The matrix 
 is approximately 60 - 70% sparse.

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in vector y

Constraints
1 ≤ M, N ≤ 10,000
The matrix 
 is approximately 60-70% sparse (i.e., 60-70% of elements are zero)
*/


__global__ void matrix_vector_multiplication_kernel(const float* A, const float* x, float* y, int M, int N) { 
    int offset = blockIdx.x * N;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += A[offset + i] * x[i];
    }
    __syncthreads();

    sum = block_all_reduce_sum(sum);
    if (threadIdx.x == 0) {
        y[blockIdx.x] = sum;
    }
}


// A, x, y are device pointers
void matrix_vector_multiplication_cuda(const float* A, const float* x, float* y, int M, int N, int nnz) {
    dim3 block(256);
    dim3 grid(M);
    matrix_vector_multiplication_kernel<<<grid, block>>>(A, x, y, M, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
} 