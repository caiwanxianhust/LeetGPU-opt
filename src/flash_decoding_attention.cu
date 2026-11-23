#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>

static constexpr int CHUNK_SIZE = 128;

static __global__ void flash_decoding_chunked_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H, int D)
{
    const int chunk_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.z;
    const int start_chunk = chunk_idx * CHUNK_SIZE;
    const int eff_chunk = min(CHUNK_SIZE, S - start_chunk);
    const int num_chunks = div_up(S, CHUNK_SIZE);

    const float* Q_ptr = Q + (query_idx * H + head_idx) * D;
    const float* K_ptr = K + (start_chunk * H + head_idx) * D;
    const float* V_ptr = V + (start_chunk * H + head_idx) * D;
    float * output_ptr = O_chunk + ((query_idx * H + head_idx) * num_chunks + chunk_idx) * D;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));

    extern __shared__ float s_buf[];
    float* s_Q = s_buf;
    // load Q into shared memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        s_Q[i] = Q_ptr[i];
    }
    __syncthreads();

    __shared__ float s_qk[CHUNK_SIZE];

    float max_score = -1e20f;
    float sum_exp_score = 0.0f;

    // comnpute qk, each warp processed one k row
    for (int row_id = warp_id; row_id < eff_chunk; row_id += (blockDim.x >> 5)) {
        float qk = 0.0f;
        for (int i = lane_id; i < D; i += 32) {
            qk += s_Q[i] * K_ptr[row_id * H * D + i];
        }
        qk = warp_all_reduce_sum(qk);
        if (lane_id == 0) {
            s_qk[row_id] = qk * softmax_scale;
        }
        __syncwarp();
    }
    __syncthreads();

    // compute max score
    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        max_score = max(max_score, s_qk[i]);
    }
    max_score = block_all_reduce_max(max_score);

    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        float exp_score = expf(s_qk[i] - max_score);
        s_qk[i] = exp_score;
        sum_exp_score += exp_score;
    }
    sum_exp_score = block_all_reduce_sum(sum_exp_score);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.0f;
        for (int row_id = 0; row_id < eff_chunk; ++row_id) {
            float weight = s_qk[row_id] / sum_exp_score;
            float v = V_ptr[row_id * H * D + i];
            acc += weight * v;
        }
        output_ptr[i] = acc;
    }

    // LSE
    if (threadIdx.x == 0) {
        const int lse_offset = query_idx * H * num_chunks + head_idx * num_chunks + chunk_idx;
        lse[lse_offset] = max_score + logf(sum_exp_score);
    }
}


static __global__ void flash_decoding_reduce_attention_kernel(
    float* __restrict__ output,
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H, int D
) {
    const int query_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_chunks = div_up(S, CHUNK_SIZE);

    float* O_ptr = output + (query_idx * H + head_idx) * D;
    float* O_chunk_ptr = O_chunk + (query_idx * H + head_idx) * num_chunks * D;
    float* lse_ptr = lse + (query_idx * H + head_idx) * num_chunks;

    extern __shared__ float s_buf[];
    float* s_lse = s_buf;               // [num_chunks,]

    float max_lse = -1e20f;
    // Phase 0: lse 整块搬进 shared（只从 global 读这一次）
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
    {
        float lse_val = lse_ptr[i];
        max_lse = max(max_lse, lse_val);
        s_lse[i] = lse_val;
    }
    __syncthreads();

    max_lse = block_all_reduce_max(max_lse);

    float deno = 0.0f;
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
    {
        deno += expf(s_lse[i] - max_lse);
    }
    deno = block_all_reduce_sum(deno);

    float global_lse = max_lse + logf(deno);

    // lse_s 原地改写成权重
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
        s_lse[i] = expf(s_lse[i] - global_lse);
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.0f;
        for (int row_id = 0; row_id < num_chunks; row_id++) {
            float w = s_lse[row_id];
            acc += O_chunk_ptr[row_id * D + i] * w;
        }
        O_ptr[i] = acc;
    }
}





static void flash_decoding_chunked_attention(
    const float* Q, const float* K, const float* V, float* output,
    float* d_O_chunk, float* d_lse,
    int T, int S, int H, int D)
{
    const int num_chunks = div_up(S, CHUNK_SIZE);

    dim3 grid(num_chunks, H, T);
    dim3 block(128);
    flash_decoding_chunked_attention_kernel<<<grid, block, D * sizeof(float)>>>(
        Q, K, V, d_O_chunk, d_lse, T, S, H, D);
    CHECK_CUDA_ERROR(cudaGetLastError());

    grid = dim3(H, T);
    block = dim3(128);
    flash_decoding_reduce_attention_kernel<<<grid, block, num_chunks * sizeof(float)>>>(
        output, d_O_chunk, d_lse, T, S, H, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


/**
 * @brief Flash decoding attention CUDA kernel launcher.
 * @param Q [T, H, D] Pointer to the query tensor on the device. 
 * @param K [S, H, D] Pointer to the key tensor on the device. 
 * @param V [S, H, D] Pointer to the value tensor on the device. 
 * @param output [T, H, D] Pointer to the output tensor on the device. 
 * @param T Length of the query sequence.
 * @param S Length of the key/value sequence.
 * @param H Number of heads.
 * @param D Dimension of each head.
 * @details Constraints
1 ≤ T ≤ 16
1 ≤ S ≤ 131,072
1 ≤ H ≤ 64
1 ≤ D ≤ 256
Input values are in [-6, 6]
Performance is measured with T = 1, S = 65,536, H = 32, D = 128

The two-pass workspace (per-chunk partial outputs + per-chunk log-sum-exp) is
allocated here from the PyTorch caching allocator instead of with cudaMalloc:
repeated calls with the same size get the same block back from the pool for
~2 us instead of paying a driver round trip of several hundred microseconds.
 */
// Q, K, V, output are device pointers
void flash_decoding_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, 
    int T, int S, int H, int D) 
{
    const int num_chunks = div_up(S, CHUNK_SIZE);
    const int64_t O_chunk_elems = static_cast<int64_t>(T) * H * num_chunks * D;
    const int64_t lse_elems = static_cast<int64_t>(T) * H * num_chunks;

    torch::Tensor scratch = torch::empty(
        {O_chunk_elems + lse_elems},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    float* d_O_chunk = scratch.data_ptr<float>();
    float* d_lse = d_O_chunk + O_chunk_elems;

    flash_decoding_chunked_attention(Q, K, V, output, d_O_chunk, d_lse, T, S, H, D);
}



torch::Tensor flash_decoding_attention(
    const torch::Tensor & Q,
    const torch::Tensor & K,
    const torch::Tensor & V
) {
    TORCH_CHECK(Q.dim() == 3, "Q must be a 3D tensor [T, H, D]");
    TORCH_CHECK(K.dim() == 3, "K must be a 3D tensor [S, H, D]");
    TORCH_CHECK(V.dim() == 3, "V must be a 3D tensor [S, H, D]");
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(K.sizes().equals(V.sizes()), "K and V must have the same shape [S, H, D]");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "Q, K and V must live on the same device");
    TORCH_CHECK(Q.size(0) >= 1 && K.size(0) >= 1, "T and S must be at least 1");
    TORCH_CHECK(Q.size(1) == K.size(1),
                "Q and K must agree on the number of heads H");
    TORCH_CHECK(Q.size(2) == K.size(2),
                "Q and K must agree on the head dimension D");
    TORCH_CHECK(Q.size(2) >= 1, "D must be at least 1");

    const int T = Q.size(0);
    const int S = K.size(0);
    const int H = K.size(1);
    const int D = K.size(2);
    torch::Tensor output = torch::empty_like(Q);

    flash_decoding_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        T, S, H, D
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}