#include "utils.cuh"

#include <torch/extension.h>


/**
Implement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU. The program should take two input vectors of equal length and produce a single output vector containing their sum.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in vector C
Example 1:
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
Example 2:
Input:  A = [1.5, 1.5, 1.5]
        B = [2.3, 2.3, 2.3]
Output: C = [3.8, 3.8, 3.8]
Constraints
Input vectors A and B have identical lengths
1 ≤ N ≤ 100,000,000
 */


__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void vector_add_naive(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


__global__ void vector_add_grid_loop_kernel(const float* A, const float* B, float* C, int N) { 
    int tid  = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}


void vector_add_grid_loop(const float* A, const float* B, float* C, int N) {
    int block_size = 256;
    int grid_size;
    if (N < block_size * 1024) {
        grid_size = (N + block_size - 1) / block_size;
    } else {
        grid_size = 1024;
    }

    vector_add_grid_loop_kernel<<<grid_size, block_size>>>(A, B, C, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void vector_add_cuda(const float* A, const float* B, float* C, int N) {
    // vector_add_naive(A, B, C, N);
    vector_add_grid_loop(A, B, C, N);
}


torch::Tensor vector_add(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have the same size");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");
    int N = A.size(0);
    auto options = A.options();
    torch::Tensor C = torch::empty({N}, options);
    vector_add_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    return C;
}