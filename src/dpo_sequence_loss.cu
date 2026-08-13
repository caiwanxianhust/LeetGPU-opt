#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>



/**
 * 软Plus 函数
 * y = log(1 + exp(x))
 */
static __device__ __host__ __forceinline__ float softplus(float x) {
    return fmaxf(0.0f, x) + log1pf(expf(-fabsf(x)));
}


static __global__ void dpo_sequence_loss_kernel(
    const float* __restrict__ chosen_logps, 
    const float* __restrict__ rejected_logps,
    const float* __restrict__ chosen_ref_logps, 
    const float* __restrict__ rejected_ref_logps,
    float* __restrict__ output, 
    float beta, int B) 
{
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    if (global_idx < B) {
        float val_chosen_logp = chosen_logps[global_idx];
        float val_rejected_logp = rejected_logps[global_idx];
        float val_chosen_ref_logp = chosen_ref_logps[global_idx];
        float val_rejected_ref_logp = rejected_ref_logps[global_idx];
        float delta_pi = val_chosen_logp - val_rejected_logp;
        float delta_ref = val_chosen_ref_logp - val_rejected_ref_logp;
        float logits_z = beta * (delta_pi - delta_ref);
        local_loss = softplus(-logits_z);
    }
    __syncthreads();
    local_loss = block_all_reduce_sum(local_loss) / (float)B;

    if (threadIdx.x == 0) {
        atomicAdd(&output[0], local_loss);
    }
}


/**
 * DPO Sequence Loss
 * @param chosen_logps: (B,)
 * @param rejected_logps: (B,)
 * @param chosen_ref_logps: (B,)
 * @param rejected_ref_logps: (B,)
 * @param output: (1,)
 */
// chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output are device pointers
void dpo_sequence_loss_cuda(
    const float* chosen_logps, const float* rejected_logps,
    const float* chosen_ref_logps, const float* rejected_ref_logps, float* output,
    float beta, int B) 
{
    dim3 block(1024);
    dim3 grid(div_up(B, block.x));
    dpo_sequence_loss_kernel<<<grid, block>>>(
        chosen_logps, rejected_logps, chosen_ref_logps, rejected_ref_logps, output, beta,B
    );
}


torch::Tensor dpo_sequence_loss(
    const torch::Tensor& chosen_logps, const torch::Tensor& rejected_logps,
    const torch::Tensor& chosen_ref_logps, const torch::Tensor& rejected_ref_logps,
    float beta
) {
    TORCH_CHECK(chosen_logps.is_cuda(), "chosen_logps must be a CUDA tensor");
    TORCH_CHECK(rejected_logps.is_cuda(), "rejected_logps must be a CUDA tensor");
    TORCH_CHECK(chosen_ref_logps.is_cuda(), "chosen_ref_logps must be a CUDA tensor");
    TORCH_CHECK(rejected_ref_logps.is_cuda(), "rejected_ref_logps must be a CUDA tensor");
    TORCH_CHECK(chosen_logps.dim() == 1, "chosen_logps must be a 1D tensor");
    TORCH_CHECK(rejected_logps.dim() == 1, "rejected_logps must be a 1D tensor");
    TORCH_CHECK(chosen_ref_logps.dim() == 1, "chosen_ref_logps must be a 1D tensor");
    TORCH_CHECK(rejected_ref_logps.dim() == 1, "rejected_ref_logps must be a 1D tensor");

    int B = chosen_logps.size(0);
    auto output = torch::zeros({1}, chosen_logps.options());

    dpo_sequence_loss_cuda(
        chosen_logps.data_ptr<float>(), rejected_logps.data_ptr<float>(),
        chosen_ref_logps.data_ptr<float>(), rejected_ref_logps.data_ptr<float>(),
        output.data_ptr<float>(), beta, B
    );

    return output;
}