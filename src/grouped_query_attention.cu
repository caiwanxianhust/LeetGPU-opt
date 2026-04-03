#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>


/*
Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during inference by sharing key and value heads across groups of query heads. Given query tensor Q with num_q_heads heads and key/value tensors K, V each with num_kv_heads heads, compute scaled dot-product attention where every group of num_q_heads / num_kv_heads consecutive query heads attends to the same key and value head. All tensors use float32.

scale = 1 / sqrt(head_dim)
scores = Q @ K^T * scale
weights = softmax(scores)
output = weights @ V

Implementation Requirements
Implement the function solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim).
Do not change the function signature or use external libraries beyond the standard GPU frameworks.
Write the result into the provided output buffer.
num_q_heads is always divisible by num_kv_heads.
Use scaled dot-product attention with scale factor 1 / sqrt(head_dim) and a softmax over the key dimension.

Constraints
1 ≤ num_kv_heads ≤ num_q_heads ≤ 64
num_q_heads is divisible by num_kv_heads
1 ≤ seq_len ≤ 4,096
8 ≤ head_dim ≤ 256; head_dim is a multiple of 8
All tensor values are float32
Performance is measured with num_q_heads = 32, num_kv_heads = 8, seq_len = 1,024, head_dim = 128
*/


namespace GQA 
{

constexpr int BLOCK_SIZE = 128;
constexpr int NUM_WARPS = BLOCK_SIZE >> 5; // 4 warps per block

__global__ void gqa_kernel(const float* Q, const float* K, const float* V, float* output,
    int num_q_heads, int num_kv_heads, int seq_len, int head_dim) { 
    const int q_head_idx = blockIdx.y;
    const int q_seq_idx = blockIdx.x;
    const int kv_head_idx = (q_head_idx * num_kv_heads) / num_q_heads;

    // ptrs to the start of the current query head, key head, and value head
    const float* q_ptr = Q + (q_head_idx * seq_len * head_dim) + (q_seq_idx * head_dim);
    const float* k_ptr = K + (kv_head_idx * seq_len * head_dim);
    const float* v_ptr = V + (kv_head_idx * seq_len * head_dim);
    float* out_ptr = output + (q_head_idx * seq_len * head_dim) + (q_seq_idx * head_dim);
    
    const int warp_id = threadIdx.x >> 5; // 0 to 7
    const int lane_id = threadIdx.x & 31; // 0 to 31

    const float scale = rsqrtf(static_cast<float>(head_dim));

    extern __shared__ float s_ptr[];
    float* s_scores = s_ptr;

    // Q @ K^T * scale
    for (int k_seq_idx = warp_id; k_seq_idx < seq_len; k_seq_idx += NUM_WARPS) {
        float tmp = 0.0f;
        for (int d = lane_id; d < head_dim; d += 32) {
            tmp += q_ptr[d] * k_ptr[k_seq_idx * head_dim + d];
        }
        __syncwarp();
        // warp reduction to sum tmp across the warp
        tmp = warp_all_reduce_sum(tmp);
        if (lane_id == 0) {
            s_scores[k_seq_idx] = tmp * scale;
        }
    }
    __syncthreads();
    
    // softmax
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = max(max_val, s_scores[i]);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    float exp_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(s_scores[i] - max_val);
        exp_sum += val;
        s_scores[i] = val;
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        s_scores[i] /= exp_sum;
    }
    __syncthreads();

    // weights @ V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int v_seq_idx = 0; v_seq_idx < seq_len; v_seq_idx++) {
            val += s_scores[v_seq_idx] * v_ptr[v_seq_idx * head_dim + d];
        }
        out_ptr[d] = val;
    }
}

// Q, K, V, output are device pointers
// Q has shape (num_q_heads, seq_len, head_dim)
// K, V have shape (num_kv_heads, seq_len, head_dim)
void gqa_cuda(const float* Q, const float* K, const float* V, float* output,
    int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(seq_len, num_q_heads);
    size_t smem_size = (seq_len) * sizeof(float);
    gqa_kernel<<<grid, block, smem_size>>>(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim);
}

} // namespace GQA
