#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>

/*
Solve the Ordinary Least Squares (OLS) regression problem on a GPU. Given a feature matrix 
 of size 
 and a target vector 
 of size 
, compute the coefficient vector 
 that minimizes the sum of squared residuals:
    minimize ||y - X * beta||^2
The closed-form solution to OLS is:
    beta = (X^T * X)^-1 * X^T * y
Implementation Requirements
External libraries are not permitted.
The solve function signature must remain unchanged.
The final coefficients must be stored in the beta vector.
Assume that the feature matrix 
 is full rank (i.e., 
 is invertible).

Constraints
1 ≤ n_samples ≤ 100,000
1 ≤ n_features ≤ 1,000
n_samples ≥ n_features
-1000.0 ≤ values in X and y ≤ 1000.0
Solutions are tested with absolute tolerance of 1e-2 and relative tolerance of 1e-2
*/





// X^T
template <int tile_size>
__global__ void mat_transpose_kernel(const float* A, float* At, int rows, int cols) {
    __shared__ float tile[tile_size][tile_size + 1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        tile[threadIdx.y][threadIdx.x] = A[row * cols + col];
    }
    __syncthreads();

    row = blockIdx.x * blockDim.x + threadIdx.y;
    col = blockIdx.y * blockDim.y + threadIdx.x;
    if (row < cols && col < rows) {
        At[row * rows + col] = tile[threadIdx.x][threadIdx.y];
    }
}

// X^T * y
// (X^T * X)^-1 * (X^T * y)
__global__ void mv_kernel(const float* xt, const float* y, float* Xty, int n) {
    int tid = blockIdx.x * n + threadIdx.x;
    int row = blockIdx.x;
    float val = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        val += xt[row * n + i] * y[i];
    }
    __syncthreads();

    val = block_all_reduce_sum(val);
    if (threadIdx.x == 0) {
        Xty[row] = val;
    }
}


// init d_xtx_inv to identity matrix
__global__ void init_identity_matrix_kernel(float* A, int n) {
    for (int i = threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        A[i * n + i] = 1.0f;
    }
}


// Gaussian elimination
__global__ void gaussian_elimination_kernel(float* A, float* AI, int n) { 
    // 循环每一列
    for (int col = 0; col < n; col++) {
        float pivot = A[col * n + col];
        // 第 col 行归一化
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            A[col * n + i] /= pivot;
            AI[col * n + i] /= pivot;
        }
        __syncthreads();

        // 每一行消元
        for (int row = 0; row < n; row++) {
            float factor = A[row * n + col];
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                if (row != col) {
                    A[row * n + i] -= factor * A[col * n + i];
                    AI[row * n + i] -= factor * AI[col * n + i];
                }
            }
        }
        __syncthreads();
    }
}


// X^T * X 
template <int tile_size>
__global__ void matrix_muliply_kernel(const float* A, const float* B, float* C, int m, int n, int k) {
    __shared__ float s_A[tile_size * tile_size];
    __shared__ float s_B[tile_size * tile_size];

    int tx = threadIdx.x % tile_size;
    int ty = threadIdx.x / tile_size;
    int row = blockIdx.y * tile_size + ty;
    int col = blockIdx.x * tile_size + tx;
    float acc = 0.0f;

    for (int i = 0; i < k; i += tile_size) {
        // load A and B from global memory to shared memory
        for (int j = threadIdx.x; j < tile_size * tile_size; j += blockDim.x) {
            int row_a = blockIdx.y * tile_size + j / tile_size;
            int col_a = i + j % tile_size;
            s_A[j] = (row_a < m && col_a < k) ? A[row_a * k + col_a] : 0.0f;
        }
        for (int j = threadIdx.x; j < tile_size * tile_size; j += blockDim.x) {
            int row_b = i + j / tile_size;
            int col_b = blockIdx.x * tile_size + j % tile_size;
            s_B[j] = (row_b < k && col_b < n) ? B[row_b * n + col_b] : 0.0f;
        }
        __syncthreads();

        if (row < m && col < n) {
            for (int kk = 0; kk < tile_size; kk++) {
                acc += s_A[ty * tile_size + kk] * s_B[kk * tile_size + tx];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}

// X, y, beta are device pointers
void ordinary_least_squares_cuda(const float* X, const float* y, float* beta, int n_samples, int n_features) {
    cudaStream_t stream_1, stream_2;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_2));

    float *d_buf;
    float *d_xt;
    float *d_xtx;
    float *d_xty;
    float *d_xtx_inv;
    size_t buf_size = (n_features * n_samples + n_features + n_features * n_features + n_features * n_features) * sizeof(float);
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_buf, buf_size, stream_1));
    d_xt = d_buf;
    d_xty = d_xt + n_features * n_samples;
    d_xtx = d_xty + n_features;
    d_xtx_inv = d_xtx + n_features * n_features;

    cudaEvent_t trans_event, xty_event, eye_init_event;
    CHECK_CUDA_ERROR(cudaEventCreate(&trans_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&xty_event));
    CHECK_CUDA_ERROR(cudaEventCreate(&eye_init_event));

    // X^T -> (n_features, n_samples)
    constexpr int tile_size = 32;
    dim3 block(tile_size, tile_size);
    dim3 grid(div_up(n_features, tile_size), div_up(n_samples, tile_size));
    mat_transpose_kernel<tile_size><<<grid, block, 0, stream_1>>>(X, d_xt, n_samples, n_features);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(trans_event, stream_1));

    // init d_xtx_inv to identity matrix
    init_identity_matrix_kernel<<<1, 256, 0, stream_2>>>(d_xtx_inv, n_features);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(eye_init_event, stream_2));

    // xtx * y -> (n_features,)
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_2, trans_event));
    mv_kernel<<<n_features, 128, 0, stream_2>>>(d_xt, y, d_xty, n_samples);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaEventRecord(xty_event, stream_2));

    // X^T * X -> (n_features, n_features)
    block = dim3(tile_size * tile_size);
    grid = dim3(div_up(n_features, tile_size), div_up(n_features, tile_size));
    matrix_muliply_kernel<tile_size><<<grid, block, 0, stream_1>>>(d_xt, X, d_xtx, n_features, n_features, n_samples);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // (X^T * X)^-1
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_1, eye_init_event));
    gaussian_elimination_kernel<<<1, 1024, 0, stream_1>>>(d_xtx, d_xtx_inv, n_features);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // (X^T * X)^-1 * (X^T * y) -> (n_features,)
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_1, xty_event));
    mv_kernel<<<n_features, 128, 0, stream_1>>>(d_xtx_inv, d_xty, beta, n_features);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_1));

    // free
    CHECK_CUDA_ERROR(cudaEventDestroy(trans_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(xty_event));
    CHECK_CUDA_ERROR(cudaEventDestroy(eye_init_event));
    CHECK_CUDA_ERROR(cudaFreeAsync(d_buf, stream_1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_2));
}


torch::Tensor ordinary_least_squares(
    const torch::Tensor& X,
    const torch::Tensor& y
) { 
    TORCH_CHECK(X.ndimension() == 2, "Input X must be 2D");
    TORCH_CHECK(y.ndimension() == 1, "Input y must be 1D");
    TORCH_CHECK(X.device().is_cuda(), "Input X must be on CUDA");
    TORCH_CHECK(y.device().is_cuda(), "Input y must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "Input X must be of type float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "Input y must be of type float32");
    TORCH_CHECK(X.size(0) == y.size(0), "Input tensors must have the same size");

    torch::Tensor beta = torch::empty({X.size(1)}, X.options());
    ordinary_least_squares_cuda(
        X.data_ptr<float>(),
        y.data_ptr<float>(),
        beta.data_ptr<float>(),
        X.size(0), X.size(1)
    );

    return beta;
}
