#include "utils.cuh"

#include "cuda_kernels.cuh"
#include <torch/extension.h>

/*
Write a GPU program that counts the number of elements with the integer value k in an 2D array of 32-bit integers. The program should count the number of elements with k in an 2D array. You are given an input 2D array input of length N x M and integer k.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: input [[1, 2, 3],
              [4, 5, 1]]
       k = 1
Output: output = 2
Example 2:
Input: input [[5, 10],
              [5, 2]]
       k = 1
Output: output = 0
Constraints
1 ≤ N, M ≤ 10,000
1 ≤ input[i], k ≤ 100
*/


__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {

    int row_id = blockIdx.x;
    int vsum = 0;

    for (int i=threadIdx.x; i<M; i+=blockDim.x) {
        vsum += (input[row_id * M + i] == K) ? 1 : 0;
    }

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void count_2d_equal_cuda(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(N);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor count_equal_2D(
    const torch::Tensor& input,
    int M, int N, int K
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) == M * N, "Input must be a 2D matrix");
    torch::Tensor output = torch::zeros({1}, input.options());
    count_2d_equal_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        M, N, K
    );
    return output;
}