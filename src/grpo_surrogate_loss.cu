#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


static __forceinline__ int div_up(int a, int b)
{
    return (a + b - 1) / b;
}


/**
 * @brief    Advantage estimation
 * @param    rewards, advantage are device pointers
 * @param    rewards [B, G]
 * @param    advantage [B, G]
 * 
 */
static __global__ void advantage_kernel(const float* __restrict__ rewards, 
    float* __restrict__ advantage, int B, int G)
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int b_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (b_id >= B) return;

    float mu = 0.0f;
    float square_sum = 0.0f;
    for (int i = lane_id; i < G; i += 32) {
        float val = rewards[b_id * G + i];
        mu += val;
        square_sum += val * val;
    }
    __syncwarp();

    mu = warp_all_reduce_sum(mu) / (float)G;
    square_sum = warp_all_reduce_sum(square_sum) / (float)G;
    float inv_std = 1.0f / (sqrtf(square_sum - mu * mu) + 1e-8f);

    for (int i = lane_id; i < G; i += 32) {
        float val = rewards[b_id * G + i];
        advantage[b_id * G + i] = (val - mu) * inv_std;
    }
}


/**
 * @brief    GRPO surrogate loss
 * @param    advantage [B, G]
 * @param    log_pi [B, G, S]
 * @param    log_pi_old [B, G, S]
 * @param    log_ref [B, G, S]
 * @param    output [1,]
 */
static __global__ void grpo_surrogate_loss_kernel(
    const float* __restrict__ advantage, 
    const float* __restrict__ log_pi, 
    const float* __restrict__ log_pi_old,
    const float* __restrict__ log_ref, 
    float* __restrict__ output, 
    float clip_eps, float beta, int B, int G, int S)
{ 
    const int b_id = blockIdx.y;
    const int g_id = blockIdx.x;
    const float val_advantage = advantage[b_id * G + g_id];
    const int offset = b_id * G * S + g_id * S;
    const float * ptr_log_pi = log_pi + offset;
    const float * ptr_log_pi_old = log_pi_old + offset;
    const float * ptr_log_ref = log_ref + offset;

    float grpo_loss = 0.0f;

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        float val_log_pi = ptr_log_pi[i];
        float val_log_pi_old = ptr_log_pi_old[i];
        float val_log_ref = ptr_log_ref[i];
        float ratio = expf(val_log_pi - val_log_pi_old);
        float clipped_ratio = min(max(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
        float surr1 = ratio * val_advantage;
        float surr2 = clipped_ratio * val_advantage;
        float ppo_loss = min(surr1, surr2);
        float d = val_log_ref - val_log_pi;
        float kl_div = exp(d) - d - 1.0f;
        grpo_loss += beta * kl_div - ppo_loss;
    }
    __syncthreads();

    float total_elem = (float)(B * G * S);
    grpo_loss = block_all_reduce_sum(grpo_loss) / total_elem;
    if (threadIdx.x == 0) {
        atomicAdd(&output[0], grpo_loss);
    }
}


static void launch_grpo_surrogate_loss_kernel(
    const float* advantage, 
    const float* log_pi, 
    const float* log_pi_old,
    const float* log_ref, 
    float* output, 
    float clip_eps, float beta, int B, int G, int S)
{
    dim3 block(256);
    dim3 grid(G, B);
    grpo_surrogate_loss_kernel<<<grid, block>>>(advantage, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S);
}


static void launch_advantage_kernel(const float* rewards, float* advantage, int B, int G)
{
    dim3 block(128);
    dim3 grid(div_up(B, 128 >> 5));
    advantage_kernel<<<grid, block>>>(rewards, advantage, B, G);
}






/**
 * @brief    GRPO surrogate loss
 * @param    rewards, log_pi, log_pi_old, log_ref, output are device pointers
 * @param    rewards [B, G]
 * @param    log_pi [B, G, S]
 * @param    log_pi_old [B, G, S]
 * @param    log_ref [B, G, S]
 * @param    output [1,]
 * 
 */
// rewards, log_pi, log_pi_old, log_ref, output are device pointers
void grpo_surrogate_loss_cuda(const float* rewards, const float* log_pi, const float* log_pi_old,
    const float* log_ref, float* output, float clip_eps, float beta, int B, int G, int S) 
{
    float * advantage;
    CHECK_CUDA_ERROR(cudaMalloc(&advantage, B * G * sizeof(float)));

    launch_advantage_kernel(rewards, advantage, B, G);

    launch_grpo_surrogate_loss_kernel(advantage, log_pi, log_pi_old, log_ref, output, clip_eps, beta, B, G, S);

    CHECK_CUDA_ERROR(cudaFree(advantage));
}


torch::Tensor grpo_surrogate_loss(
    const torch::Tensor& rewards,
    const torch::Tensor& log_pi,
    const torch::Tensor& log_pi_old,
    const torch::Tensor& log_ref,
    float clip_eps,
    float beta
) {
    TORCH_CHECK(rewards.ndimension() == 2, "rewards must be 2D");
    TORCH_CHECK(log_pi.ndimension() == 3, "log_pi must be 3D");
    TORCH_CHECK(log_pi_old.ndimension() == 3, "log_pi_old must be 3D");
    TORCH_CHECK(log_ref.ndimension() == 3, "log_ref must be 3D");

    int B = rewards.size(0);
    int G = rewards.size(1);
    int S = log_pi.size(2);
    TORCH_CHECK(log_pi.size(0) == B && log_pi.size(1) == G, "log_pi must have shape (B, G, S)");
    TORCH_CHECK(log_pi_old.size(0) == B && log_pi_old.size(1) == G, "log_pi_old must have shape (B, G, S)");
    TORCH_CHECK(log_ref.size(0) == B && log_ref.size(1) == G, "log_ref must have shape (B, G, S)");

    torch::Tensor output = torch::empty({1}, rewards.options());
    grpo_surrogate_loss_cuda(
        rewards.data_ptr<float>(),
        log_pi.data_ptr<float>(),
        log_pi_old.data_ptr<float>(),
        log_ref.data_ptr<float>(),
        output.data_ptr<float>(),
        clip_eps, beta, B, G, S
    );
    return output;
}
