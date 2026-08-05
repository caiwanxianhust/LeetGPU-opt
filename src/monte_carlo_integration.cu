#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>

/* 
Implement Monte Carlo integration on a GPU. Given a set of function values 
 sampled at random points 
 uniformly distributed in the interval 
, estimate the definite integral:

The Monte Carlo method approximates the integral by computing the average of the function values and multiplying by the interval width.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the result variable
Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2
Example:
Input:  a = 0, b = 2, n_samples = 8
        y_samples = [0.0625, 0.25, 0.5625, 1.0, 1.5625, 2.25, 3.0625, 4.0]
Output: result = 3.1875
Constraints
1 ≤ n_samples ≤ 100,000,000
-1000.0 ≤ a < b ≤ 1000.0
-10000.0 ≤ function values ≤ 10000.0
The tolerance is set to 1e-2 to account for the inherent randomness in Monte Carlo methods and floating-point precision variations.
*/


static __device__ __host__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void reduce_in_block_kernel(const float* y_samples, float* result_in_block, float scaler, int n_samples) {
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n_samples; i += gridDim.x * blockDim.x) {
        sum += y_samples[i];
    }
    __syncthreads();

    sum = block_all_reduce_sum(sum);
    if (threadIdx.x == 0) {
        result_in_block[blockIdx.x] = scaler * sum;
    }
}

// y_samples, result are device pointers
void monte_carlo_integration_cuda(const float* y_samples, float* result, float a, float b, int n_samples) {
    int block_size = 1024;
    int grid_size;
    if (n_samples <= block_size) {
        grid_size = 1;
    }
    else if (1024 * block_size >= n_samples) {
        grid_size = div_up(n_samples, block_size);
    }
    else {
        grid_size = 1024;
    }

    float scaler = (b - a) / static_cast<float>(n_samples);
    float *d_result_in_block;
    CHECK_CUDA_ERROR(cudaMalloc(&d_result_in_block, grid_size * sizeof(float)));
    reduce_in_block_kernel<<<grid_size, block_size>>>(y_samples, d_result_in_block, scaler, n_samples);
    CHECK_CUDA_ERROR(cudaGetLastError());
    reduce_in_block_kernel<<<1, 1024>>>(d_result_in_block, result, 1.0f, grid_size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_result_in_block));
}


torch::Tensor monte_carlo_integration(
    const torch::Tensor& y_samples,
    float a, float b
) {
    TORCH_CHECK(y_samples.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(y_samples.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(y_samples.dtype() == torch::kFloat32, "Input must be of type float32");

    int N = y_samples.size(0);
    torch::Tensor result = torch::empty({1}, y_samples.options());
    monte_carlo_integration_cuda(
        y_samples.data_ptr<float>(),
        result.data_ptr<float>(),
        a, b, N
    );
    return result;
}