#include "utils.cuh"

#include <cfloat>
#include <torch/extension.h>
#include <tuple>

/*
MoE Top-K Gating
Medium
Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape [M, E] where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the topk_weights and topk_indices arrays
Example 1:
Input:
  logits = [[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]]
  M = 2, E = 4, k = 2

Output:
  topk_weights = [[0.2689, 0.7311],
                  [0.7311, 0.2689]]
  topk_indices = [[2, 3],
                  [0, 1]]

Explanation:
Row 0: Top-2 values are 3.0 and 4.0 at indices 2 and 3.
       Softmax([3.0, 4.0]) = [0.2689, 0.7311]
Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
       Softmax([4.0, 3.0]) = [0.7311, 0.2689]
Constraints
1 ≤ M ≤ 10,000 (number of tokens)
1 ≤ E ≤ 256 (number of experts)
1 ≤ k ≤ E (top-k selection, typically k=2)
All tensors are stored on GPU
Logits are 32-bit floats
Indices are 32-bit integers
Performance is measured with M = 1,024, k = 2
*/


static __host__ __device__ __forceinline__ int div_up(int x, int y) {
    return (x + y - 1) / y;
}


static __device__ __forceinline__ void insert_to_min_heap(float *heap, int *heap_idx, float val, int idx, int k) {
    if (val < heap[k - 1]) return;
    int i = k - 2;
    while (i >= 0 && heap[i] < val) {
        heap[i + 1] = heap[i];
        heap_idx[i + 1] = heap_idx[i];
        i--;
    }
    heap[i + 1] = val;
    heap_idx[i + 1] = idx;
}



__global__ void moe_topk_gating_kernel(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row_id = blockIdx.x * 4 + warp_id;
    if (row_id >= M) return;
    
    // find topk values
    __shared__ float s_min_heap[4][256];
    __shared__ int s_min_heap_idx[4][256];
    __shared__ float s_buf[4][256];
    __shared__ float sum_exp[4];

    for (int i = lane_id; i < E; i += 32) {
        s_buf[warp_id][i] = logits[row_id * E + i];
        s_min_heap[warp_id][i] = -FLT_MAX;
        s_min_heap_idx[warp_id][i] = -1;
    }
    __syncwarp();


    if (lane_id == 0) {
        int heap_size = 0;
        for (int i = 0; i < E; i++) {
            heap_size = heap_size < k ? heap_size + 1 : k;
            insert_to_min_heap(s_min_heap[warp_id], s_min_heap_idx[warp_id], s_buf[warp_id][i], i, heap_size);
            // // heap not full
            // if (heap_size < k) {
            //     heap_size++;
            //     insert_to_min_heap(s_min_heap[warp_id], s_min_heap_idx[warp_id], s_buf[warp_id][i], i, heap_size);
            // } else {
            //     insert_to_min_heap(s_min_heap[warp_id], s_min_heap_idx[warp_id], s_buf[warp_id][i], i, k);
            // }
        }

        float max_val = s_min_heap[warp_id][0];
        sum_exp[warp_id] = 0.0f;
        for (int i = 0; i < k; i++) {
            float exp_val = expf(s_min_heap[warp_id][i] - max_val);
            s_min_heap[warp_id][i] = exp_val;
            sum_exp[warp_id] += exp_val;
        }
    }
    __syncwarp();

    // store weights & indices in gmem
    for (int i = lane_id; i < k; i += 32) {
        topk_indices[row_id * k + i] = s_min_heap_idx[warp_id][i];
        topk_weights[row_id * k + i] = s_min_heap[warp_id][i] / sum_exp[warp_id];
    }
}



// logits, topk_weights, topk_indices are device pointers
void moe_topk_gating_cuda(const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k) {
    // each warp process one row
    constexpr int BLOCK_SIZE = 128;
    int grid_size = div_up(M, 128 / 32);
    moe_topk_gating_kernel<<<grid_size, BLOCK_SIZE>>>(logits, topk_weights, topk_indices, M, E, k);
}


std::tuple<torch::Tensor, torch::Tensor> moe_topk_gating(const torch::Tensor& logits, int k)
{ 
    TORCH_CHECK(logits.ndimension() == 2, "Logits must be 2D");
    TORCH_CHECK(logits.device().is_cuda(), "Logits must be on CUDA");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Logits must be of type float32");
    TORCH_CHECK(k > 0 && k <= logits.size(1), "Invalid k");
    int M = logits.size(0);
    int E = logits.size(1);
    torch::Tensor topk_weights = torch::empty({M, k}, logits.options());
    torch::Tensor topk_indices = torch::empty({M, k}, logits.options().dtype(torch::kInt32));
    moe_topk_gating_cuda(
        logits.data_ptr<float>(), 
        topk_weights.data_ptr<float>(), 
        topk_indices.data_ptr<int>(), 
        M, E, k);
    return {topk_weights, topk_indices};
}