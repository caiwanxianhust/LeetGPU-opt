#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Write a GPU program that counts the number of elements with the integer value p in an 3D array of 32-bit integers. The program should count the number of elements with p in an 3D array. You are given an input 3D array input of length N x M x K and integer p.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: input [[[1, 2, 3],
               [4, 5, 1]],
              [[1, 1, 1],
               [2, 2, 2]]]
       N = 2, M = 2, K = 3
       p = 1
Output: output = 5
Example 2:
Input: input [[[5, 10],
               [5, 2],
               [2, 2]]]
       N = 1, M = 3, K = 2 
       p = 1
Output: output = 0
Constraints
1 ≤ N, M, K ≤ 1,000
1 ≤ input[i], p ≤ 100
*/


// static __host__ __device__ __forceinline__ int div_up(int a, int b) {
//     return (a + b - 1) / b;
// }




__global__ void count_3d_equal_kernel(const int* input, int* output, int N, int M, int K) {

    int row_id = blockIdx.x;
    int vsum = 0;

    for (int i=threadIdx.x; i<M; i+=blockDim.x) {
        vsum += (input[row_id * M + i] == K) ? 1 : 0;
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
void count_array_element_3D_cuda(const int* input, int* output, int N, int M, int K, int P) {
    dim3 block(256);
    dim3 grid(N * M);
    count_3d_equal_kernel<<<grid, block>>>(input, output, N * M, K, P);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor count_equal_3D(
    const torch::Tensor& input,
    const int n, const int m, const int k,
    const int p
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) == n * m * k, "Input must be a 3D matrix");

    torch::Tensor output = torch::zeros({1}, input.options());

    count_array_element_3D_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, m, k, p
    );

    return output;
}