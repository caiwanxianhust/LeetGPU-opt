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


__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float vsum = 0.0f;
    if (row < M && col < K) {
        for (int i=0; i<N; ++i) {
            vsum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = vsum;
    }


}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void matrix_multiplication_cuda(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
