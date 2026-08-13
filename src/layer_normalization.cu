#include "utils.cuh"
#include "cuda_kernels.cuh"


#include <cuda_runtime.h>
#include <torch/extension.h>


static __global__ void layer_normalization_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C, float eps) 
{ 
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int batch_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (batch_id >= N) return;

    float mean = 0.0f;
    float square_sum = 0.0f;
    for (int i = lane_id; i < C; i += 32) {
        float val = input[batch_id * C + i];
        mean += val;
        square_sum += val * val;
    }
    __syncwarp();
    mean = warp_all_reduce_sum(mean) / (float)C;
    square_sum = warp_all_reduce_sum(square_sum) / (float)C;
    float inv_std = rsqrtf(square_sum - mean * mean + eps);

    for (int i = lane_id; i < C; i += 32) {
        float val = (input[batch_id * C + i] - mean) * inv_std;
        output[batch_id * C + i] = val * weight[i] + bias[i];
    }
}


static __global__ void layer_normalization_dp4_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output,
    int N, int C, float eps) 
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int batch_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (batch_id >= N) return;

    const float4* ptr_inp = reinterpret_cast<const float4*>(input + batch_id * C);
    const float4* ptr_weight = reinterpret_cast<const float4*>(weight);
    const float4* ptr_bias = reinterpret_cast<const float4*>(bias);
    float4* ptr_out = reinterpret_cast<float4*>(output + batch_id * C);

    const int C4 = C >> 2;
    float mean = 0.0f;
    float square_sum = 0.0f;
    for (int i = lane_id; i < C4; i += 32) { 
        float4 val = ptr_inp[i];
        mean += val.x + val.y + val.z + val.w;
        square_sum += (val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w);
    }
    __syncwarp();

    mean = warp_all_reduce_sum(mean) / (float)C;
    square_sum = warp_all_reduce_sum(square_sum) / (float)C;
    float inv_std = rsqrtf(square_sum - mean * mean + eps);

    for (int i = lane_id; i < C4; i += 32) {
        float4 val = ptr_inp[i];
        float4 out;
        out.x = (val.x - mean) * inv_std * ptr_weight[i].x + ptr_bias[i].x;
        out.y = (val.y - mean) * inv_std * ptr_weight[i].y + ptr_bias[i].y;
        out.z = (val.z - mean) * inv_std * ptr_weight[i].z + ptr_bias[i].z;
        out.w = (val.w - mean) * inv_std * ptr_weight[i].w + ptr_bias[i].w;
        ptr_out[i] = out;
    }

}


static void launch_layer_normalization_kernel(const float* input, const float* weight, const float* bias, float* output,
    int N, int C, float eps) 
{
    constexpr int block_size = 128;
    int grid_size = div_up(N, block_size >> 5);
    if (C & 3 == 0) {
        layer_normalization_dp4_kernel<<<grid_size, block_size>>>(input, weight, bias, output, N, C, eps);
    }
    else {
        layer_normalization_kernel<<<grid_size, block_size>>>(input, weight, bias, output, N, C, eps);
    } 
}



// input, weight, bias, output are device pointers
void layer_normalization_cuda(const float* input, const float* weight, const float* bias, float* output,
    int N, int C, float eps) 
{
    launch_layer_normalization_kernel(input, weight, bias, output, N, C, eps);
}


torch::Tensor layer_normalization(
    torch::Tensor& input, 
    torch::Tensor& weight, 
    torch::Tensor& bias, 
    float eps
) { 
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(weight.size(0) == input.size(1), "weight and input must have the same size");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias and input must have the same size");

    int N = input.size(0);
    int C = input.size(1);

    torch::Tensor output = torch::empty_like(input);
    layer_normalization_cuda(
        input.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(),
        N, C, eps);

    return output;
}