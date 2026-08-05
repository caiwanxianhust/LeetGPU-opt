#include "utils.cuh"

#include <torch/extension.h>

/*
Implement the SiLU (Sigmoid Linear Unit) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  input = [0.5, 1.0, -0.5]  (N=3)
Output: output = [0.3112295, 0.731059, -0.1887705]
Example 2:
Input:  input = [-1.0, -2.0, -3.0, -4.0, -5.0]  (N=5)
Output: output = [-0.26894143 -0.23840584 -0.14227763 -0.07194484 -0.03346425]
Constraints
1 ≤ N ≤ 10,000
-100.0 ≤ input values ≤ 100.0
*/

__device__ __forceinline__  float silu(float val) {
    return val / (1 + __expf(-1.0f * val));
}

__global__ void silu_kernel(const float* input, float* output, int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        output[tid] = silu(input[tid]);
    }

}

// input, output are device pointers
void silu_cuda(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor silu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    silu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}