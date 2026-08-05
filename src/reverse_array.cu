#include "utils.cuh"

#include <torch/extension.h>


/*
Reverse Array
Easy
Implement a program that reverses an array of 32-bit floating point numbers in-place. The program should perform an in-place reversal of input.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored back in input
Example 1:
Input: [1.0, 2.0, 3.0, 4.0]
Output: [4.0, 3.0, 2.0, 1.0]
Example 2:
Input: [1.5, 2.5, 3.5]
Output: [3.5, 2.5, 1.5]
Constraints
1 ≤ N ≤ 100,000,000
*/


__global__ void reverse_array_kernel(float* input, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ridx = N - 1 - idx;
    int half_n = (N - 1) / 2 + 1;
    if (idx < half_n) {
        float tmp = input[idx];
        input[idx] = input[ridx];
        input[ridx] = tmp;
    }
}

// input is device pointer
void reverse_array_cuda(float* input, int N) {
    int threadsPerBlock = 256;
    int half_n = (N - 1) / 2 + 1;
    int blocksPerGrid = (half_n + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}


void reverse_array(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    reverse_array_cuda(
        input.data_ptr<float>(),
        N
    );
    return;
}