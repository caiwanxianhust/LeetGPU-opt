#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <assert.h>


/*
Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape [N, C] where N is the batch size and C is the number of features, compute the normalized output using learnable scale (gamma) and shift (beta) parameters.

For each feature channel j, batch normalization computes:
 
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output tensor
Example 1:
Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (N=3, C=2)
        gamma = [1.0, 1.0]
        beta = [0.0, 0.0]
        eps = 1e-5
Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]
Example 2:
Input:  input = [[0.0, 1.0], [2.0, 3.0]]  (N=2, C=2)
        gamma = [2.0, 0.5]
        beta = [1.0, -1.0]
        eps = 1e-5
Output: output = [[-1.0, -1.5], [3.0, -0.5]]
Constraints
1 ≤ N ≤ 10,000
1 ≤ C ≤ 1,024
eps = 1e-5
-100.0 ≤ input values ≤ 100.0
0.1 ≤ gamma values ≤ 10.0
-10.0 ≤ beta values ≤ 10.0
*/


__global__ void batch_norm_kernel(const float* input, const float* gamma, const float* beta, 
                                  float* output, int N, int C, float eps) {
    if (threadIdx.x < C) {
        int feature_channel = threadIdx.x;
        float mean = 0.0f;
        float mean_squared = 0.0f;
        float val;

        for (int i = 0; i < N; i++) {
            val = input[i * C + feature_channel];
            mean += val;
            mean_squared += val * val;
        }
        mean /= N;
        mean_squared /= N;
        float variance = mean_squared - mean * mean;
        float std_dev = sqrtf(variance + eps);

        for (int i = 0; i < N; i++) {
            val = input[i * C + feature_channel];
            output[i * C + feature_channel] = (val - mean) / std_dev * gamma[feature_channel] + beta[feature_channel];
        }
    }
}



// input, gamma, beta, output are device pointers
void batch_norm_cuda(const float* input, const float* gamma, const float* beta, 
                     float* output, int N, int C, float eps) {
    assert(C <= 1024);
    batch_norm_kernel<<<1, 1024>>>(input, gamma, beta, output, N, C, eps);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}