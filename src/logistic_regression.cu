#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>

/*
Solve the logistic regression problem on a GPU. Given a feature matrix 
 of size 
 and a binary target vector 
 of size 
 (containing only 0s and 1s), compute the coefficient vector 
 that maximizes the log-likelihood:
 
 \beta = argmax_\beta \sum_{i=1}^n [y_i log(1 + exp(-X_i^T \beta)) + (1 - y_i) log(1 + exp(X_i^T \beta))]

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final coefficients must be stored in the beta vector
The target vector y contains only binary values (0 and 1)

Constraints
1 ≤ n_samples ≤ 100,000
1 ≤ n_features ≤ 1,000
n_samples ≥ n_features
-10.0 ≤ values in X ≤ 10.0
y contains only binary values: 0 or 1
Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2
*/


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static __device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


// L2 init Loss
__global__ void l2_init_loss_kernel(
    const float* beta, float* loss, int n_samples, int n_features, float lambda
) {
    float val = 0.0f;
    for (int i=threadIdx.x; i < n_features; i += blockDim.x) {
        val += beta[i] * beta[i];
    }
    __syncthreads();

    val = block_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        loss[0] = lambda * val / (2 * static_cast<float>(n_samples));
    }
}


// z = sigmoid( X * beta )
// loss = -1 / n * ( sum(y * log(z) + (1 - y) * log(1 - z))  +    )
__global__ void mv_sigmoid_loss_kernel(
    const float* X, const float* y, const float* beta, 
    float* output, float* loss, int n_samples, int n_features
) {
    int row  = blockIdx.x;
    float val = 0.0f;
    for (int i=threadIdx.x; i < n_features; i += blockDim.x) {
        val += X[row * n_features + i] * beta[i];
    }
    __syncthreads();

    val = block_all_reduce_sum(val);
    val = sigmoid(val);
    if (threadIdx.x == 0) {
        output[row] = val;
        float loss_patial = -(y[row] * logf(val) + (1.0f - y[row]) * logf(1.0f - val)) / static_cast<float>(n_samples);
        atomicAdd(&loss[0], loss_patial);
    }
}


template <int tile_size>
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float s_buf[tile_size][tile_size + 1];
    int col  = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) { 
        s_buf[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    __syncthreads();

    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    if (row < cols && col < rows) {
        output[row * cols + col] = s_buf[threadIdx.x][threadIdx.y];
    }
}

// grad = 1 / n * ( X.T * (z - y) + lambda * beta)
// w = w - alpha · grad
__global__ void lr_gradient_update_kernel(const float* XT, const float* y, const float* z, float* beta, int n, float lambda, float alpha) { 
    int feature_id = blockIdx.x;
    float val = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        val += XT[feature_id * n + i] * (z[i] - y[i]);
    }
    __syncthreads();
    val = block_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        float grad = 1.0f / static_cast<float>(n) * (val + lambda * beta[feature_id]);
        beta[feature_id] = beta[feature_id] - alpha * grad;
    }
}


__global__ void lr_update_kernel(const float* grad, float* beta, float alpha, int n_features) {
    int feature_id = blockIdx.x;
    if (threadIdx.x == 0) {
        beta[feature_id] -= alpha * grad[feature_id];
    }
}



// X, y, beta are device pointers
void logistic_regression_cuda(const float* X, const float* y, float* beta, int n_samples, int n_features) {

    float* d_buf;
    float* d_z;
    float* d_xt;
    float* d_grad;
    float* d_loss;

    size_t buf_size = (n_samples + n_features * n_samples + n_features + 1) * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    d_z = d_buf;
    d_xt = d_z + n_samples;
    d_grad = d_xt + n_features * n_samples;
    d_loss = d_grad + n_features;

    float eps = 1e-6f;
    int epochs = 50;
    float lambda = 0.01f;
    float alpha = 0.001f;

    float prev_loss = -1000.0f;
    float curr_loss;

    // X.T
    constexpr int tile_size = 32;
    dim3 block(tile_size, tile_size);
    dim3 grid(div_up(n_features, tile_size), div_up(n_samples, tile_size));
    matrix_transpose_kernel<tile_size><<<grid, block>>>(X, d_xt, n_samples, n_features);
    CHECK_CUDA_ERROR(cudaGetLastError());

    for (int i = 0; i < epochs; i++) { 
        // init loss
        l2_init_loss_kernel<<<1, 256>>>(beta, d_loss, n_samples, n_features, lambda);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // z = sigmoid( X * beta )
        // loss = -1 / n * sum(y * log(z) + (1 - y) * log(1 - z))
        mv_sigmoid_loss_kernel<<<n_samples, 256>>>(X, y, beta, d_z, d_loss, n_samples, n_features);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // grad = 1 / n * ( X.T * (z - y) + lambda * beta)
        // w = w - alpha · grad
        lr_gradient_update_kernel<<<n_features, 256>>>(d_xt, y, d_z, beta, n_samples, lambda, alpha);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemcpy(&curr_loss, d_loss, sizeof(float) * 1, cudaMemcpyDeviceToHost));
        // printf("Epoch: %d, prev Loss: %f Current Loss: %f\n", i, prev_loss, curr_loss);
        if (i > 0 && abs(curr_loss - prev_loss) < eps) {
            printf("Early stop at epoch: %d, current loss: %f\n", i, curr_loss);
            break;
        }
        prev_loss = curr_loss;
    }
    
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


torch::Tensor logistic_regression(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& beta
) { 
    TORCH_CHECK(X.ndimension() == 2, "Input X must be 2D");
    TORCH_CHECK(y.ndimension() == 1, "Input y must be 1D");
    TORCH_CHECK(beta.ndimension() == 1, "Input beta must be 1D");
    TORCH_CHECK(X.device().is_cuda(), "Input X must be on CUDA");
    TORCH_CHECK(y.device().is_cuda(), "Input y must be on CUDA");
    TORCH_CHECK(beta.device().is_cuda(), "Input beta must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "Input X must be of type float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "Input y must be of type float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "Input beta must be of type float32");
    TORCH_CHECK(X.size(0) == y.size(0), "Input tensors must have the same size");
    TORCH_CHECK(X.size(1) == beta.size(0), "Input tensors must have compatible shapes");
    logistic_regression_cuda(
        X.data_ptr<float>(),
        y.data_ptr<float>(),
        beta.data_ptr<float>(),
        X.size(0), X.size(1)
    );
    return beta;
}