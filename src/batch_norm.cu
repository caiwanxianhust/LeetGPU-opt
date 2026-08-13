#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <assert.h>
#include <torch/extension.h>


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




namespace batch_norm_naive {

__global__ void batch_norm_naive_kernel(const float* input, const float* gamma, const float* beta,
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
void launch_batch_norm_naive_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps) {
    assert(C <= 1024);
    batch_norm_naive_kernel << <1, 1024 >> > (input, gamma, beta, output, N, C, eps);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

}   // namespace batch_norm_naive



namespace batch_norm_tile {

constexpr int TILE_SIZE = 32;

__global__ void batch_norm_tile_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps) {
    __shared__ float s_mean[TILE_SIZE][TILE_SIZE];
    __shared__ float s_mean_squared[TILE_SIZE][TILE_SIZE];
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < C) {
        float mean = 0.0f;
        float mean_squared = 0.0f;
        float val;

        for (int i = threadIdx.y; i < N; i+=blockDim.y){
            val = input[i * C + col];
            mean += val;
            mean_squared += val * val;
        }
        s_mean[threadIdx.y][threadIdx.x] = mean;
        s_mean_squared[threadIdx.y][threadIdx.x] = mean_squared;
        __syncthreads();

        // reduce mean and mean_squared in the tile
        if (threadIdx.y == 0) {
            mean = 0;
            mean_squared = 0;
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                mean += s_mean[i][threadIdx.x];
                mean_squared += s_mean_squared[i][threadIdx.x];
            }
            mean /= N;
            mean_squared /= N;
            float scale = 1.0f / sqrtf(mean_squared - mean * mean + eps);
            // Store the scale in shared memory, refuse to store the mean and variance to save shared memory
            s_mean[0][threadIdx.x] = mean;
            s_mean_squared[0][threadIdx.x] = scale;
        }
        __syncthreads();

        float gamma_val = gamma[col];
        float beta_val = beta[col];
        for (int i = threadIdx.y; i < N; i+=blockDim.y) { 
            output[i * C + col] = (input[i * C + col] - s_mean[0][threadIdx.x]) * s_mean_squared[0][threadIdx.x] * gamma_val + beta_val; 
        }
    }
}


void launch_batch_norm_tile_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(div_up(C, TILE_SIZE), 1);
    batch_norm_tile_kernel << <grid, block >> > (input, gamma, beta, output, N, C, eps);
}

}   // namespace batch_norm_tile


namespace batch_norm_transpose { 

constexpr int TILE_SIZE = 32;

__global__ void mat_transpose_kernel(const float* input, float* output, int M, int N) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    __shared__ float s_tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    s_tile[ty][tx] = (row < M && col < N) ? input[row * N + col] : 0.0f;
    __syncthreads();

    row = bx * TILE_SIZE + ty; // transpose block offset
    col = by * TILE_SIZE + tx;
    if (row < N && col < M) {
        output[row * M + col] = s_tile[tx][ty]; // write transposed
    }
}

void launch_mat_transpose_kernel(const float* input, float* output, int M, int N) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(M, TILE_SIZE));
    mat_transpose_kernel<<<grid, block >>>(input, output, M, N);
}


/**
 * layer_norm_kernel
 * @param input: [C, N]
 * @param gamma: [C]
 * @param beta: [C]
 * @param output: [C, N]
 * @param N: batch size
 * @param C: feature channels
 */
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps) { 

    const int feature_channel = blockIdx.x;
    float gamma_val = gamma[feature_channel];
    float beta_val = beta[feature_channel];

    float mean = 0.0f;
    float mean_squared = 0.0f;
    float val;
    for (int i = threadIdx.x; i < N; i+=blockDim.x) {
        val = input[feature_channel * N + i];
        mean += val;
        mean_squared += val * val;
    }
    __syncthreads();
    mean = block_all_reduce_sum(mean);
    mean /= N;
    mean_squared = block_all_reduce_sum(mean_squared);
    mean_squared /= N;
    float std_dev = 1.0f / sqrtf(mean_squared - mean * mean + eps);

    for (int i = threadIdx.x; i < N; i+=blockDim.x) {
        output[feature_channel * N + i] = (input[feature_channel * N + i] - mean) * std_dev * gamma_val + beta_val;
    }
}

void launch_layer_norm_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps) {
    dim3 block(256);
    dim3 grid(C);
    layer_norm_kernel<<<grid, block >>>(input, gamma, beta, output, N, C, eps);
}


void launch_batch_norm_transpose_kernel(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps)
{
    float * d_buf;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, 2 * C * N * sizeof(float)));
    float * transpose_input = d_buf;
    float * layer_output = transpose_input + C * N;
    launch_mat_transpose_kernel(input, transpose_input, N, C);
    launch_layer_norm_kernel(transpose_input, gamma, beta, layer_output, N, C, eps);
    launch_mat_transpose_kernel(layer_output, output, C, N);

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


}   // namespace batch_norm_transpose



void batch_norm_cuda(const float* input, const float* gamma, const float* beta,
    float* output, int N, int C, float eps)
{
    // batch_norm_tile::launch_batch_norm_tile_kernel(input, gamma, beta, output, N, C, eps);
    batch_norm_transpose::launch_batch_norm_transpose_kernel(input, gamma, beta, output, N, C, eps);
}


torch::Tensor batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(gamma.ndimension() == 1, "Gamma must be 1D");
    TORCH_CHECK(beta.ndimension() == 1, "Beta must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(gamma.device().is_cuda(), "Gamma must be on CUDA");
    TORCH_CHECK(beta.device().is_cuda(), "Beta must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "Gamma must be of type float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "Beta must be of type float32");
    TORCH_CHECK(input.size(1) == gamma.size(0), "Input and gamma must have compatible shapes");
    TORCH_CHECK(input.size(1) == beta.size(0), "Input and beta must have compatible shapes");

    int N = input.size(0);
    int D = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({N, D}, options);

    batch_norm_cuda(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D, eps
    );

    return output;
}