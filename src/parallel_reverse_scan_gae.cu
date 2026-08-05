#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>



static __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


static __global__ void naive_reverse_scan_gae_kernel(
    const float* __restrict__ rewards, 
    const float* __restrict__ values, 
    float* __restrict__ advantages, 
    float gamma, float lam, int B, int S) 
{ 
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float c = gamma * lam;
    if (tid < B) {
        for (int t = S - 1; t >= 0; t--) {
            float r = rewards[tid * S + t];
            float v = values[tid * S + t];
            float v_next = (t == S - 1) ? 0.0f : values[tid * S + t + 1];
            float delta = r + gamma * v_next - v;
            advantages[tid * S + t] = (t == S - 1) ? delta : advantages[tid * S + t + 1] * c + delta;
        }
    }
}


static void launch_naive_reverse_scan_gae_kernel(
    const float* __restrict__ rewards, 
    const float* __restrict__ values, 
    float* __restrict__ advantages, 
    float gamma, float lam, int B, int S) 
{
    constexpr int BLOCK_SIZE = 256;
    naive_reverse_scan_gae_kernel<<<div_up(B, BLOCK_SIZE), BLOCK_SIZE>>>(
        rewards, values, advantages, gamma, lam, B, S);
}

// ============================================================
// 融合 Delta 计算 + 分块串行扫描 GAE
// ============================================================
#define CHUNK_SIZE 1024

__global__ void fused_chunk_scan_gae_kernel(
    const float* __restrict__ rewards,
    const float* __restrict__ values,
    float* __restrict__ advantages,
    float* __restrict__ chunk_carries,
    float gamma, float c, int B, int S, int num_chunks) 
{
    int chunk_idx = blockIdx.x;
    if (chunk_idx >= num_chunks) return;

    int batch_idx = blockIdx.y;
    if (batch_idx >= B) return;

    int tid = threadIdx.x;
    int start = chunk_idx * CHUNK_SIZE;
    int end = min(start + CHUNK_SIZE, S);
    int len = end - start;

    // 使用共享内存进行块内串行计算
    __shared__ float s_adv[CHUNK_SIZE];

    // 1. 所有线程并行将 rewards 和 values 倒序加载到共享内存
    // 这里我们直接加载原始数据，不再单独存 delta
    int global_pos = (tid < len) ? (end - 1 - tid) : 0;
    float r = (tid < len) ? rewards[batch_idx * S + global_pos] : 0.0f;
    float v = (tid < len) ? values[batch_idx * S + global_pos] : 0.0f;
    
    __shared__ float s_v_next[CHUNK_SIZE]; // 需要额外存一下 v_next 用于算 delta
    s_v_next[tid] = (tid < len) ? ((global_pos == S - 1) ? 0.0f : values[batch_idx * S + global_pos + 1]) : 0.0f;
    
    // 直接在共享内存中算出 delta 存入 s_adv，避免全局内存中转
    s_adv[tid] = r + gamma * s_v_next[tid] - v;
    __syncthreads();

    // 2. 仅 tid == 0 的线程执行串行递推
    if (tid == 0) {
        for (int i = 1; i < len; i++) {
            s_adv[i] = s_adv[i] + c * s_adv[i - 1];
        }
    }
    __syncthreads();

    // 3. 所有线程并行将结果正序写回全局内存 (advantages)
    if (tid < len) {
        int write_pos = end - 1 - tid;
        advantages[batch_idx * S + write_pos] = s_adv[tid];
    }

    // 4. 计算该 Chunk 的总进位 (Carry)
    if (tid == 0) {
        float last_val = (len > 0) ? s_adv[len - 1] : 0.0f;
        chunk_carries[batch_idx * num_chunks + chunk_idx] = last_val * c;
    }
}

// ============================================================
// 跨块进位 Kernel 
// ============================================================
__global__ void apply_carries_kernel(
    float* __restrict__ advantages,
    const float* __restrict__ chunk_carries,
    float c, int B, int S, int num_chunks) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * S) return;

    int b = idx / S;
    int t = idx % S;
    int chunk_idx = t / CHUNK_SIZE;

    if (chunk_idx >= num_chunks - 1) return;

    float incoming_carry = chunk_carries[b * num_chunks + chunk_idx + 1];

    int offset_in_chunk = t % CHUNK_SIZE;
    int chunk_len = min(CHUNK_SIZE, S - chunk_idx * CHUNK_SIZE);
    int dist_to_end = chunk_len - 1 - offset_in_chunk;

    float decay = powf(c, (float)dist_to_end);
    advantages[idx] += incoming_carry * decay;
}

// ============================================================
// Host 启动函数
// ============================================================
void launch_parallel_gae(
    const float* rewards,
    const float* values,
    float* advantages,
    float gamma, float lam, int B, int S) 
{
    float c = gamma * lam;
    int num_chunks = div_up(S, CHUNK_SIZE);
    int total_elements = B * S;

    // 分配 chunk_carries 这一个极小的临时缓冲区！
    float *d_chunk_carries;
    cudaMalloc(&d_chunk_carries, B * num_chunks * sizeof(float));

    // 1. 融合 Kernel：计算 Delta + 块内串行扫描
    dim3 scan_blocks(num_chunks, B);
    fused_chunk_scan_gae_kernel<<<scan_blocks, CHUNK_SIZE>>>(
        rewards, values, advantages, d_chunk_carries, gamma, c, B, S, num_chunks);

    // 2. 跨块进位叠加
    int threads = 256;
    int blocks = div_up(total_elements, threads);
    apply_carries_kernel<<<blocks, threads>>>(
        advantages, d_chunk_carries, c, B, S, num_chunks);

    // 清理临时显存
    cudaFree(d_chunk_carries);
}


/**
 * @brief 跨块逆序扫描gae
 * @param rewards 奖励  [B, S]
 * @param values 价值   [B, S]
 * @param advantages 优势   [B, S]
 * @param gamma 折扣因子
 * @param lambda gae系数
 * @param B 批量大小
 * @param S 状态维度
 */
// rewards, values, advantages are device pointers
void parallel_reverse_scan_gae_cuda(const float* rewards, const float* values, 
    float* advantages, float gamma, float lam, int B, int S) 
{
    launch_parallel_gae(rewards, values, advantages, gamma, lam, B, S);
}


torch::Tensor parallel_reverse_scan_gae(
    torch::Tensor rewards, 
    torch::Tensor values, 
    float gamma, float lam) 
{ 
    TORCH_CHECK(rewards.is_cuda(), "rewards must be a CUDA tensor");
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(rewards.ndimension() == 2, "rewards must be a 2D tensor")
    TORCH_CHECK(values.ndimension() == 2, "values must be a 2D tensor")

    int B = rewards.size(0);
    int S = rewards.size(1);

    torch::Tensor advantages = torch::zeros_like(rewards);
    parallel_reverse_scan_gae_cuda(
        rewards.data_ptr<float>(),
        values.data_ptr<float>(),
        advantages.data_ptr<float>(),
        gamma, lam, B, S
    );
    return advantages;
}