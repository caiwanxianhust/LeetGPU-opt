#include "utils.cuh"

#include <torch/extension.h>


/*
Write a GPU program that counts the number of elements with the integer value k in an array of 32-bit integers. The program should count the number of elements with k in an array. You are given an input array input of length N and integer k.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: [1, 2, 3, 4, 1], k = 1
Output: 2
Example 2:
Input: [5, 10, 5, 2], k = 11
Output: 0
Constraints
1 ≤ N ≤ 100,000,000
1 ≤ input[i], k ≤ 100,000
*/


__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    __shared__ int s_cnt;
    if (threadIdx.x == 0) {
        s_cnt = 0;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        if (input[tid] == K) atomicAdd(&s_cnt, 1);
    }
    __syncthreads();

    if (threadIdx.x == 1) {
        atomicAdd(output, s_cnt);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void count_equal_cuda(const int* input, int* output, int N, int K) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor count_equal(
    const torch::Tensor& input,
    int K
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    int N = input.size(0);
    torch::Tensor output = torch::zeros({1}, input.options());
    count_equal_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        N,
        K
    );
    return output;
}