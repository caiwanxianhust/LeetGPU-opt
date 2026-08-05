#include "utils.cuh"

#include <torch/extension.h>


/*
Implement a program that performs the Rectified Linear Unit (ReLU) activation function on a vector of 32-bit floating point numbers. The ReLU function sets all negative values to zero and leaves positive values unchanged:

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in output
Example 1:
Input:  input = [-2.0, -1.0, 0.0, 1.0, 2.0]
Output: output = [0.0, 0.0, 0.0, 1.0, 2.0]
Example 2:
Input:  input = [-3.5, 0.0, 4.2]
Output: output = [0.0, 0.0, 4.2]
Constraints
1 ≤ N ≤ 100,000,000
*/

__global__ void relu_kernel(const float* input, float* output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = fmax(0.0f, input[tid]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void relu_cuda(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}



torch::Tensor relu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    relu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}