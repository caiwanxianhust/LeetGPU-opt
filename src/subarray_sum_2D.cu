#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Implement a program that computes the sum of a 2D subarray of 32-bit integers. You are given an input 2D array input of length N x M, and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL. S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input:  input = [[1, 2, 3],
                 [4, 5, 1]]

        S_ROW = 0, E_ROW = 1, S_COL = 1, E_COL = 2
Output: output = 11
Example 2:
Input:  input = [[5, 10],
                 [5, 2]]
        S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 1
Output: output = 10
Constraints
1 ≤ N, M ≤ 10,000
1 ≤ input[i] ≤ 10
0 ≤ S_ROW ≤ E_ROW ≤ N - 1
0 ≤ S_COL ≤ E_COL ≤ M - 1
*/


static __global__ void zeros_init_kernel(int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = 0;
    }
}

__global__ void subarray_sum_2D_kernel(
    const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int row_id = blockIdx.x + S_ROW;
    if (row_id > E_ROW) {
        return;
    }
    int col_id = threadIdx.x + S_COL;
    int vsum = 0;
    for (int i = col_id; i <= E_COL; i+= blockDim.x) {
        vsum += input[row_id * M + i];
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
void subarray_sum_2D_cuda(const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    dim3 block(256);
    dim3 grid(E_ROW - S_ROW + 1);
    zeros_init_kernel<<<1, 256>>>(output, 1);
    subarray_sum_2D_kernel<<<grid, block>>>(input, output, N, M, S_ROW, E_ROW, S_COL, E_COL);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor subarray_sum_2D(
    const torch::Tensor& input,
    const int S_ROW, const int E_ROW, const int S_COL, const int E_COL
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(E_ROW >= S_ROW && input.size(0) >= E_ROW, "Invalid range");
    TORCH_CHECK(E_COL >= S_COL && input.size(1) >= E_COL, "Invalid range");

    int n = input.size(0);
    int m = input.size(1);
    
    torch::Tensor output = torch::zeros({1}, input.options());
    subarray_sum_2D_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, m,
        S_ROW, E_ROW, S_COL, E_COL
    );
    return output;
}