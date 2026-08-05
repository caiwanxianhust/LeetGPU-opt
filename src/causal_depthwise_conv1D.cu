#include "utils.cuh"

#include <torch/extension.h>


/*
Implement a causal depthwise 1D convolution over a batched sequence tensor x of shape (B, L, D), producing an output of the same shape. In a depthwise convolution, each channel d is convolved independently using its own kernel weight[d, :] — there is no mixing across channels. The convolution is causal: output position l may only depend on input positions 0, 1, …, l (past and present), never future positions. This operation is a key component of state-space models such as Mamba, where it is applied before the selective scan to mix local context within each feature channel.

where positions l − k < 0 are treated as zero (zero-pad the left boundary). The tensor layout is channels-last: x[b, l, d] is stored at offset b × L × D + l × D + d.

Implementation Requirements
The solve function signature must remain unchanged
The result must be written into the output tensor
Use only native features (external libraries are not permitted)
Input positions before the start of the sequence (i.e. indices l − k < 0) must be treated as zero
Example
With B = 1, L = 4, D = 2, K = 3:

x      = [[[1.0, 2.0],    # l=0
           [3.0, 4.0],    # l=1
           [5.0, 6.0],    # l=2
           [7.0, 8.0]]]   # l=3   shape (1, 4, 2)

weight = [[ 1.0,  0.0, -1.0],   # channel d=0
          [ 1.0,  1.0,  1.0]]   # channel d=1   shape (2, 3)

bias   = [0.0, 0.0]

output = [[[1.0,  2.0],   # l=0: d0: 1*1=1          d1: 1*2=2
           [3.0,  6.0],   # l=1: d0: 3*1+1*0=3      d1: 4*1+2*1=6
           [4.0, 12.0],   # l=2: d0: 5*1+3*0+1*(-1)=4  d1: 6+4+2=12
           [4.0, 18.0]]]  # l=3: d0: 7*1+5*0+3*(-1)=4  d1: 8+6+4=18
Constraints
1 ≤ B ≤ 16 (batch size)
1 ≤ L ≤ 8,192 (sequence length)
1 ≤ D ≤ 8,192 (number of channels)
1 ≤ K ≤ 8 (kernel size; typically 3 or 4 in practice)
All tensors use 32-bit floating point
Tensor x and output use channels-last layout: shape (B, L, D)
Performance is measured with B = 8, L = 2,048, D = 4,096, K = 4
*/


namespace naive {

__global__ void causal_depthwise_conv1D_naive_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output, 
    int B, int L, int D, int K)
{
    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;

    extern __shared__ float s_buf[];
    float* s_x = s_buf;
    float* s_weight = s_buf + L;

    for (int seq_id = threadIdx.x; seq_id < L; seq_id += blockDim.x) {
        s_x[seq_id] = x[batch_id * L * D + seq_id * D + channel_id];
    }
    for (int k_id = threadIdx.x; k_id < K; k_id += blockDim.x) {
        s_weight[k_id] = weight[channel_id * K + k_id];
    }
    __syncthreads();

    for (int seq_id = threadIdx.x; seq_id < L; seq_id += blockDim.x) {
        float val = bias[channel_id];
        for (int k = 0; k < K && k <= seq_id; k++) {
            val += weight[channel_id * K + k] * s_buf[seq_id - k];
        }
        output[batch_id * L * D + seq_id * D + channel_id] = val;
    }

}


void launch_causal_depthwise_conv1D_naive(
    const float* x, const float* weight, const float* bias, float* output, 
    int B, int L, int D, int K)
{
    dim3 grid(B, D);
    dim3 block(1024);
    size_t smem_size = (K + L) * sizeof(float);
    causal_depthwise_conv1D_naive_kernel<<<grid, block, smem_size>>>(
        x, weight, bias, output, B, L, D, K);
}


}   // namespace naive


namespace r87 {
__global__ void causal_depthwise_conv1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B,
    int L,
    int D,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * L * D;

    if (idx >= total) return;

    int d = idx % D;
    int tmp = idx / D;
    int l = tmp % L;
    int b = tmp / L;

    float acc = bias[d];

    int base_x = (b * L + l) * D + d;
    int base_w = d * K;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (k < K && l >= k) {
            acc += weight[base_w + k] * x[base_x - k * D];
        }
    }

    output[idx] = acc;
}

void launch_causal_depthwise_conv1d_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    int B,
    int L,
    int D,
    int K
) {
    int total = B * L * D;

    constexpr int threads = 256;
    int blocks = (total + threads - 1) / threads;

    causal_depthwise_conv1d_kernel<<<blocks, threads>>>(
        x, weight, bias, output, B, L, D, K
    );
}

}   // namespace r87




// x, weight, bias, output are device pointers
void causal_depthwise_conv1D_cuda(const float* x, const float* weight, const float* bias, float* output, int B,
    int L, int D, int K) 
{
    // naive::launch_causal_depthwise_conv1D_naive(x, weight, bias, output, B, L, D, K);
    r87::launch_causal_depthwise_conv1d_kernel(x, weight, bias, output, B, L, D, K);
}


torch::Tensor causal_depthwise_conv1D(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias
) {
    TORCH_CHECK(x.ndimension() == 3, "Input must be 3D");
    TORCH_CHECK(weight.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(bias.ndimension() == 1, "Input must be 1D");

    // 校验 D
    TORCH_CHECK(x.size(2) == weight.size(0) && x.size(2) == bias.size(0), "The ndimension of d_ffn must match");

    int B = x.size(0);
    int L = x.size(1);
    int D = x.size(2);
    int K = weight.size(1);

    torch::Tensor output = torch::zeros_like(x);

    causal_depthwise_conv1D_cuda(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L, D, K
    );

    return output;
}



