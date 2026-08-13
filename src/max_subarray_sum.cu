#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Implement a program that computes the maximum sum of any contiguous subarray of length exactly window_size. You are given an array input of length N consisting of 32-bit signed integers, and an integer window_size.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input:  input = [1, 2, 4, 2, 3], window_size = 2
Output: output = 6
Example 2:
Input:  input = [-1, -4, -2, 1], window_size = 3
Output: output = -5
Constraints
1 ≤ N ≤ 50,000
-10 ≤ input[i] ≤ 10
1 ≤ window_size ≤ N
*/





__global__ void max_subarray_sum_kernel(const int* input, int* max_in_block, int N, int window_size, int output_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vmax = -INT_MAX;
    int start_idx = tid;
    int end_idx = tid + window_size;
    if (tid < output_size && end_idx <= N) {
        vmax = 0;
        for (int i = start_idx; i < end_idx; i++) {
            vmax += input[i];
        }
    }
    __syncthreads();    

    vmax = block_all_reduce_max(vmax);
    if (threadIdx.x == 0) {
        max_in_block[blockIdx.x] = vmax;
    }
}


__global__ void reduce_max_kernel(int* max_in_block, int* output, int num_blocks) { 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vmax = -INT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x * gridDim.x) {
        vmax = max(max_in_block[i], vmax);
    }
    __syncthreads();
    vmax = block_all_reduce_max(vmax);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = vmax;
    }
}




// input, output are device pointers (i.e. pointers to memory on the GPU)
void max_subarray_sum_cuda(const int* input, int* output, int N, int window_size) {
    int output_size = N - window_size + 1;
    int block_size = 256;
    int num_blocks = div_up(output_size, block_size);
    int *d_buf;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, num_blocks * sizeof(int)));
    max_subarray_sum_kernel<<<num_blocks, block_size>>>(input, d_buf, N, window_size, output_size);
    reduce_max_kernel<<<1, 1024>>>(d_buf, output, num_blocks);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


torch::Tensor max_subarray_sum(
    const torch::Tensor& input,
    const int window_size
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(window_size > 0 && window_size <= input.size(0), "Invalid window size");

    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({1}, options);
    max_subarray_sum_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        N,
        window_size
    );
    return output;
}
