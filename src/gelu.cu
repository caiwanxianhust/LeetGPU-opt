#include "utils.cuh"

#include <torch/extension.h>




static __global__ void gelu_kernel(const float* X, float* Y, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = X[idx];
        Y[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}



// X, Y are device pointers (i.e. pointers to memory on the GPU)
static void launch_gelu_kernel(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
}


static __global__ void gelu_float4_kernel(const float* X, float* Y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N / 4) {
        float4 x = reinterpret_cast<const float4*>(X)[idx];
        float4 y;
        y.x = 0.5f * x.x * (1.0f + tanhf(0.7978845608028654f * (x.x + 0.044715f * x.x * x.x * x.x)));
        y.y = 0.5f * x.y * (1.0f + tanhf(0.7978845608028654f * (x.y + 0.044715f * x.y * x.y * x.y)));
        y.z = 0.5f * x.z * (1.0f + tanhf(0.7978845608028654f * (x.z + 0.044715f * x.z * x.z * x.z)));
        y.w = 0.5f * x.w * (1.0f + tanhf(0.7978845608028654f * (x.w + 0.044715f * x.w * x.w * x.w)));
        reinterpret_cast<float4*>(Y)[idx] = y;
    }
}


static void launch_gelu_float4_kernel(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N / 4 + threadsPerBlock - 1) / threadsPerBlock;

    gelu_float4_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
}


void gelu_cuda(const float* X, float* Y, int N) {
    if (N % 4 == 0) {
        launch_gelu_float4_kernel(X, Y, N);
    } else {
        launch_gelu_kernel(X, Y, N);
    }
}


torch::Tensor gelu(
    const torch::Tensor& X
) { 
    TORCH_CHECK(X.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(X.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "Input must be of type float32");

    int N = X.numel();
    auto Y = torch::empty_like(X);

    gelu_cuda(X.data_ptr<float>(), Y.data_ptr<float>(), N);
    return Y;
}