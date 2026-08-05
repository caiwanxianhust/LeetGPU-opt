#include "utils.cuh"

#include <torch/extension.h>


/*
Implement a program that copies an 
 matrix of 32-bit floating point numbers from input array 
 to output array 
 on the GPU. The program should perform a direct element-wise copy so that 
 for all valid indices.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in matrix B
Example 1:
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
Example 2:
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Constraints
1 ≤ N ≤ 4096
All elements are 32-bit floating point numbers
*/

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N * N) {
        B[tid] = A[tid];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
void copy_matrix_cuda(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
} 


torch::Tensor matrix_copy(
    const torch::Tensor& A,
    int N
) {
    TORCH_CHECK(A.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(A.size(0) == N * N, "Input must be square matrix");
    auto options = A.options();
    torch::Tensor B = torch::empty({N * N}, options);
    copy_matrix_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        N
    );
    return B;
}