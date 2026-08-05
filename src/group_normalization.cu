#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>



// Group Normalization CUDA Kernel
__global__ void group_norm_kernel(
    const float* __restrict__ X,   // 输入张量 (N, C, H, W)
    const float* __restrict__ gamma, // 缩放参数 (C,)
    const float* __restrict__ beta,  // 偏移参数 (C,)
    float* __restrict__ Y,         // 输出张量 (N, C, H, W)
    int N, int C, int H, int W,    // 张量维度
    int G,                         // 组数
    float eps                      // 数值稳定性参数
){
    const int batch_id = blockIdx.x;
    const int group_id = blockIdx.y;
    int group_size = C / G;
    int elements_per_group = group_size * H * W; // 每组总元素数

    float mean = 0.0f;
    float var = 0.0f;

    // cal mean
    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        mean += X[batch_id * C * H * W + group_id * group_size * H * W + i];
    }
    __syncthreads();

    mean = block_all_reduce_sum(mean);
    mean /= elements_per_group;

    // cal var
    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        float x = X[batch_id * C * H * W + group_id * group_size * H * W + i] - mean;
        var += x * x;
    }
    __syncthreads();

    var = block_all_reduce_sum(var);
    var /= elements_per_group;
    var = rsqrtf(var + eps); // 计算方差的平方根并加上eps以避免除零

    // cal normalized output
    for (int i = threadIdx.x; i < elements_per_group; i += blockDim.x) {
        int c = group_id * group_size + (i / (H * W));
        float x = X[batch_id * C * H * W + group_id * group_size * H * W + i];
        float x_hat = (x - mean) * var;
        Y[batch_id * C * H * W + group_id * group_size * H * W + i] = x_hat * gamma[c] + beta[c];
    }
}


// X, gamma, beta, Y are device pointers
void group_normalization_cuda(const float* X, const float* gamma, const float* beta, float* Y, int N, int C,
    int H, int W, int G, float eps) 
{
    dim3 block(256);
    dim3 grid(N, G);
    group_norm_kernel<<<grid, block>>>(X, gamma, beta, Y, N, C, H, W, G, eps);
}


torch::Tensor group_normalization(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int G, float eps
) {
    TORCH_CHECK(x.ndimension() == 4, "Input must be 4D");
    TORCH_CHECK(gamma.ndimension() == 1, "gamma must be 1D");
    TORCH_CHECK(beta.ndimension() == 1, "beta must be 1D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(gamma.size(0) == C, "gamma must have shape (C)");
    TORCH_CHECK(beta.size(0) == C, "beta must have shape (C)");
    TORCH_CHECK(C % G == 0, "C must be divisible by G");

    torch::Tensor output = torch::empty_like(x);
    group_normalization_cuda(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, G, eps
    );
    return output;
}