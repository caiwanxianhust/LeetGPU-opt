#include "utils.cuh"

#include <torch/extension.h>

/*
Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the output using the elementwise formula. The input and output tensor must be of type float32.

Compute the SWiGLU output:
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  [1.0, 2.0, 3.0, 4.0]  (N=4)
Output: [2.1931758, 7.0463767]
Example 2:
Input:  [0.5, 1.0]  (N=4)
Output: [0.31122968]
Constraints
1 ≤ N ≤ 100,000
N is an even number
-100.0 ≤ input values ≤ 100.0
*/

__device__ __forceinline__ float swiglu(float x1, float x2) {
    return  x1 * x2 / (1 + __expf(-1.0f * x1));
}

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < halfN) {
        output[tid] = swiglu(input[tid], input[tid + halfN]);
    }
}

// input, output are device pointers
void swiglu_cuda(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor swiglu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(input.size(0) % 2 == 0, "Input size must be even");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N / 2}, options);
    swiglu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}