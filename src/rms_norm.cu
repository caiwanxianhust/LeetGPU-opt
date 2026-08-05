#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <assert.h>
#include <torch/extension.h>

/*
Implement RMS Normalization forward pass for 1D input vectors. Given an input tensor of shape [N] where N is the number of elements, compute the normalized output using a scalar scale (gamma) and shift (beta) parameter.

RMS Normalization computes:

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  input = [1.0, 2.0, 3.0, 4.0]  (N=4)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.36514813, 0.73029625, 1.0954444, 1.4605925 ]
Example 2:
Input:  input = [1.0, 2.0, 3.0]  (N=3)
        gamma = 1.0
        beta = 0.0
        eps = 1e-5
Output: output = [0.46290955, 0.9258191, 1.3887286]
Constraints
1 ≤ N ≤ 100,000
eps = 1e-5
-100.0 ≤ input values ≤ 100.0
0.1 ≤ gamma ≤ 10.0
-10.0 ≤ beta ≤ 10.0
*/


__global__ void rms_norm_kernel(const float* input, float gamma, float beta, 
                                 float* output, int N, float eps) {
    assert(gridDim.x == 1); // only support 1 block for now
    double mean = 0.0;
    float val;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = input[i];
        mean += (double)val * (double)val;
    }
    __syncthreads();

    mean = block_all_reduce_sum(mean);
    mean = mean / N;
    mean = rsqrt(mean + (double)eps);

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = input[i];
        output[i] = gamma * (float)mean * val + beta;
    }
}


// input, output are device pointers
void rms_norm_cuda(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {
    rms_norm_kernel<<<1, 1024>>>(input, gamma, beta, output, N, eps);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor rms_norm_1D(
    const torch::Tensor& input,
    float eps, float gamma, float beta
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    rms_norm_cuda(
        input.data_ptr<float>(),
        gamma,
        beta,
        output.data_ptr<float>(),
        N,
        eps
    );
    return output;
}