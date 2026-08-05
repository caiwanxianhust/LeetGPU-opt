#include "cuda_kernels.cuh"
#include "utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void token_embedding_layer_kernel(
    const int* token_ids, const int* position_ids, const float* token_embeddings,
    const float* position_embeddings, const float* gamma, const float* beta,
    float* output, int B, int T, int V, int P, int D, float eps)
{
    const int batch_id = blockIdx.y;    // batch index
    const int seq_id = blockIdx.x;      // sequence index
    const int token_id = token_ids[batch_id * T + seq_id];
    const int pos_id = position_ids[seq_id];

    __shared__ float s_embedding[1024];

    float mean = 0.0f;
    float mean_sq = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = token_embeddings[token_id * D + i] + position_embeddings[pos_id * D + i];
        s_embedding[i] = val;
        mean += val;
        mean_sq += val * val;
    }
    __syncthreads();

    mean = block_all_reduce_sum(mean) / D;
    mean_sq = block_all_reduce_sum(mean_sq) / D;
    float var = rsqrtf(mean_sq - mean * mean + eps);

    int offset = batch_id * T * D + seq_id * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = s_embedding[i];
        output[offset + i] = (val - mean) * var * gamma[i] + beta[i];
    }
}





/**
 * token_embedding_layer_cuda
 * @brief
 * @param token_ids: [B, T] int32
 * @param position_ids: [T] int32
 * @param token_embeddings: [V, D] float32
 * @param position_embeddings: [P, D] float32
 * @param gamma: [D] float32
 * @param beta: [D] float32
 * @param output: [B, T, D] float32
 * @param B: batch size
 * @param T: sequence length
 * @param V: vocabulary size
 * @param P: max position embeddings
 * @param D: embedding dimension
 * @param eps: epsilon for layer normalization
 */
// token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta, output are device
// pointers
void token_embedding_layer_cuda(
    const int* token_ids, const int* position_ids, const float* token_embeddings,
    const float* position_embeddings, const float* gamma, const float* beta,
    float* output, int B, int T, int V, int P, int D, float eps) 
{
    dim3 block(256);
    dim3 grid(T, B);
    token_embedding_layer_kernel<<<grid, block>>>(
        token_ids, position_ids, token_embeddings, position_embeddings, gamma, beta,
        output, B, T, V, P, D, eps);

}


torch::Tensor token_embedding_layer(
    const torch::Tensor& token_ids,
    const torch::Tensor& position_ids,
    const torch::Tensor& token_embeddings,
    const torch::Tensor& position_embeddings,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps
) { 
    int B = token_ids.size(0);
    printf("B: %d\n", B);
    int T = token_ids.size(1);
    int V = token_embeddings.size(0);
    int P = position_embeddings.size(0);
    int D = token_embeddings.size(1);

    torch::Tensor output = torch::empty({B, T, D}, token_embeddings.options());
    printf("B: %d, T: %d, V: %d, P: %d, D: %d\n", B, T, V, P, D);
    
    token_embedding_layer_cuda(
        token_ids.data_ptr<int>(),
        position_ids.data_ptr<int>(),
        token_embeddings.data_ptr<float>(),
        position_embeddings.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, V, P, D, eps);

    return output;
}
