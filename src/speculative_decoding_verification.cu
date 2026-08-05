#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cuda_runtime.h>


__device__ int accept_token_idx(const int* batch_draft_tokens, const float* batch_draft_probs, 
    const float* batch_target_probs, const float* batch_uniform_samples, int* batch_output_tokens, 
    int T, int V)
{
    for (int t = 0; t < T; ++t) {
        int draft_token = batch_draft_tokens[t];
        float draft_prob = batch_draft_probs[t * V + draft_token];
        float target_prob = batch_target_probs[t * V + draft_token];
        float alpha = min(1.0f, target_prob / draft_prob);
        float u = batch_uniform_samples[t];
        if (u < alpha) {
            batch_output_tokens[t] = draft_token;
        }
        else {
            return t;
        }
    }
    return T;
}



/**
 * Speculative Decoding Verification
 * @param draft_tokens: [B, T] int32
 * @param draft_probs: [B, T, V] float32
 * @param target_probs: [B, T, V] float32
 * @param uniform_samples: [B, T + 1] float32
 * @param output_tokens: [B, T + 1] int32
 * @param B: batch size
 * @param T: max sequence length
 * @param V: vocabulary size
 */
__global__ void decoding_verify_kernel(const int* draft_tokens, const float* draft_probs, const float* target_probs,
    const float* uniform_samples, int* output_tokens, int B, int T, int V)
{
    const int batch_id = blockIdx.x;
    const int* batch_draft_tokens = draft_tokens + batch_id * T;
    const float* batch_draft_probs = draft_probs + batch_id * T * V;
    const float* batch_target_probs = target_probs + batch_id * T * V;
    const float* batch_uniform_samples = uniform_samples + batch_id * (T + 1);
    int* batch_output_tokens = output_tokens + batch_id * (T + 1);

    __shared__ int sample_id;
    if (threadIdx.x == 0) {
        sample_id = accept_token_idx(batch_draft_tokens, batch_draft_probs, batch_target_probs, batch_uniform_samples, 
                                    batch_output_tokens, T, V);
    }
    __syncthreads();
    // 所有的 token 都被接受了
    if (sample_id == T) {
        if (threadIdx.x == 0) {
            float r = batch_uniform_samples[T];
            const float* q_last = batch_target_probs + (T - 1) * V;
            float cdf_cum = 0.0f;
            int bonus_token = V - 1;
            for (int v = 0; v < V; ++v) {
                cdf_cum += q_last[v];
                if (r <= cdf_cum) {
                    bonus_token = v;
                    break;
                }
            }
            batch_output_tokens[T] = bonus_token;
        }
        return;
    }

    // 将第 sample_id 个 token 之后的所有 token 设置为 0
    for (int t = threadIdx.x + sample_id + 1; t < T + 1; ++t) {
        batch_output_tokens[t] = 0;
    }
    __syncthreads();

    // 采样确定第 sample_id 个 token
    float adj_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float val = batch_target_probs[sample_id * V + v] - batch_draft_probs[sample_id * V + v];
        adj_sum += max(0.0f, val);
    }
    __syncthreads();

    adj_sum = block_all_reduce_sum(adj_sum);

    if (threadIdx.x == 0) {
        float r = batch_uniform_samples[T];

        if (adj_sum <= 1e-20f) {
            batch_output_tokens[sample_id] = min((int)(r * V), V - 1);
            return;
        }

        float cdf_cum = 0.0f;
        for (int v = 0; v < V; ++v) {
            float val = batch_target_probs[sample_id * V + v] - batch_draft_probs[sample_id * V + v];
            val = max(0.0f, val) / adj_sum;
            cdf_cum += val;
            if (r <= cdf_cum) {
                batch_output_tokens[sample_id] = v;
                break;
            }
        }
    }

}



// draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens are device pointers
void speculative_decoding_verification_cuda(const int* draft_tokens, const float* draft_probs, const float* target_probs,
                      const float* uniform_samples, int* output_tokens, int B, int T, int V) 
{
    dim3 blockDim(256);
    dim3 gridDim(B);
    decoding_verify_kernel<<<gridDim, blockDim>>>(draft_tokens, draft_probs, target_probs, uniform_samples, output_tokens, B, T, V);
    CHECK_CUDA_ERROR(cudaGetLastError());
}
