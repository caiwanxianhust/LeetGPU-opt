#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>


static __global__ void ppo_clipped_surrogate_loss_kernel(
    const float* __restrict__ advantages, 
    const float* __restrict__ log_pi, 
    const float* __restrict__ log_pi_old,
    float* __restrict__ output, 
    float clip_eps, int B, int S)
{
    const int b_id = blockIdx.x;

    float surrogate_loss_per_token = 0.0f;
    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        const int idx = b_id * S + i;
        const float adv = advantages[idx];
        const float ratio = expf(log_pi[idx] - log_pi_old[idx]);
        const float clipped_ratio = fmaxf(fminf(ratio, 1.0f + clip_eps), 1.0f - clip_eps);
        const float loss = fminf(ratio * adv, clipped_ratio * adv);
        surrogate_loss_per_token += -loss;
    }
    __syncthreads();

    float total_elem = (float)(B * S);
    surrogate_loss_per_token = block_all_reduce_sum(surrogate_loss_per_token) / total_elem;
    if (threadIdx.x == 0) {
        atomicAdd(output, surrogate_loss_per_token);
    }
}

static void launch_ppo_clipped_surrogate_loss_kernel(
    const float* advantages, 
    const float* log_pi, 
    const float* log_pi_old,
    float* output, 
    float clip_eps, int B, int S)
{ 
    dim3 block(256);
    dim3 grid(B);
    ppo_clipped_surrogate_loss_kernel<<<grid, block>>>(
        advantages, log_pi, log_pi_old, output, clip_eps, B, S);
}



/**
 * Computes the PPO clipped surrogate loss.
 * @param    advantages [B, S]
 * @param    log_pi [B, S]
 * @param    log_pi_old [B, S]
 * @param    output [1,]
 */
// advantages, log_pi, log_pi_old, output are device pointers
void ppo_clipped_surrogate_loss_cuda(
    const float* advantages, const float* log_pi, const float* log_pi_old,
    float* output, float clip_eps, int B, int S) 
{
    launch_ppo_clipped_surrogate_loss_kernel(
        advantages, log_pi, log_pi_old, output, clip_eps, B, S);
}


torch::Tensor ppo_clipped_surrogate_loss(
    const torch::Tensor& advantages,
    const torch::Tensor& log_pi,
    const torch::Tensor& log_pi_old,
    float clip_eps)
{ 
    TORCH_CHECK(advantages.is_cuda(), "advantages must be a CUDA tensor");
    TORCH_CHECK(log_pi.is_cuda(), "log_pi must be a CUDA tensor");
    TORCH_CHECK(log_pi_old.is_cuda(), "log_pi_old must be a CUDA tensor");
    TORCH_CHECK(advantages.dim() == 2, "advantages must be a 2D tensor");
    TORCH_CHECK(log_pi.dim() == 2, "log_pi must be a 2D tensor");
    TORCH_CHECK(log_pi_old.dim() == 2, "log_pi_old must be a 2D tensor");
    TORCH_CHECK(advantages.sizes() == log_pi.sizes(), "advantages and log_pi must have the same shape");
    TORCH_CHECK(advantages.sizes() == log_pi_old.sizes(), "advantages and log_pi_old must have the same shape");

    int B = advantages.size(0);
    int S = advantages.size(1);

    torch::Tensor output = torch::zeros({1}, advantages.options());
    ppo_clipped_surrogate_loss_cuda(
        advantages.data_ptr<float>(),
        log_pi.data_ptr<float>(),
        log_pi_old.data_ptr<float>(),
        output.data_ptr<float>(),
        clip_eps,
        B,
        S
    );
    return output;
}
