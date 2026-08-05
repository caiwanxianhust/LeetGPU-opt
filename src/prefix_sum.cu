#include "utils.cuh"

#include <torch/extension.h>


/*
Write a GPU program that computes the prefix sum (cumulative sum) of an array of 32-bit floating point numbers. For an input array [a, b, c, d, ...], the prefix sum is [a, a+b, a+b+c, a+b+c+d, ...].

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The result must be stored in the output array
Example 1:
Input: [1.0, 2.0, 3.0, 4.0]
Output: [1.0, 3.0, 6.0, 10.0]
Example 2:
Input: [5.0, -2.0, 3.0, 1.0, -4.0]
Output: [5.0, 3.0, 6.0, 7.0, 3.0]
Constraints
1 ≤ N ≤ 100,000,000
-1000.0 ≤ input[i] ≤ 1000.0
The largest value in the output array will fit within a 32-bit float
*/


__global__ void prefix_sum_in_block_kernel(float* input, float* output, float* reduce_sum_output, int N) {
    __shared__ float s_data[1024];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s_data[threadIdx.x] = (tid < N) ? input[tid] : 0.0f;
    __syncthreads();

    float temp = s_data[threadIdx.x];
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        if (threadIdx.x >= offset) {
            temp += s_data[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            s_data[threadIdx.x] = temp;
        }
    }

    if (tid < N) {
        if (threadIdx.x == blockDim.x - 1) {
            reduce_sum_output[blockIdx.x] = s_data[threadIdx.x];
        }
        output[tid] = s_data[threadIdx.x];
    }
}


__global__ void add_reduce_sum_kernel(float* output, float* reduce_sum_output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = (tid < N) ? output[tid] : 0.0f;
    if (tid < N) {
        for (int i = 0; i < blockIdx.x; i++) {
            temp += reduce_sum_output[i];
        }
        output[tid] = temp;
    }
}


void prefix_sum_cuda(float* input, float* output, float* reduce_sum_output, int N) {
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    prefix_sum_in_block_kernel<<<grid_size, block_size>>>(input, output, reduce_sum_output, N);
    add_reduce_sum_kernel<<<grid_size, block_size>>>(output, reduce_sum_output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor prefix_sum(
    const torch::Tensor& input
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    torch::Tensor output = torch::empty({N}, input.options());
    torch::Tensor reduce_sum_output = torch::empty({N}, input.options());
    prefix_sum_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reduce_sum_output.data_ptr<float>(),
        N
    );
    return output;
}