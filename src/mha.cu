#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>

namespace {
    constexpr int TILE_SIZE = 16;

    __host__ __device__ int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
}


/**
 * @brief    Multi-head QK scaled
 * @param    Q, K, qk are device pointers, Q[N, h, size_per_head], K[N, h, size_per_head], qk[h, N, N]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void multi_head_qk_scaled_kernel(const float* Q, const float*K, float* qk, int N, int size_per_head, int d_model) 
{
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int offset_c = head_id * N * N;
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < size_per_head; k += TILE_SIZE) {
        // load Q and K to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_q[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? Q[row * d_model + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_k[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? K[row * d_model + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < N) {
        qk[offset_c + row_c * N + col_c] = acc / sqrtf((float)size_per_head);
    }
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[h, N, N], attn_scores[h, N, N]
 * @param    N sequence length
 * @return   void
 */
__global__ void softmax_kernel(const float* qk, float* attn_scores, int N) 
{
    float max_val = -INFINITY;
    float exp_sum = 0.0f;
    float val;

    // Find max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    // Compute exp sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        exp_sum += expf(val - max_val);
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    // Write output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        attn_scores[blockIdx.x * N + i] = expf(val - max_val) / exp_sum;
    }
}


/**
 * @brief    Multi-head attention
 * @param    attn_scores, V, output are device pointers, attn_scores[h, N, N], V[N, h, size_per_head], output[N, h, size_per_head]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void mha_kernel(const float* attn_scores, const float* V, float* output, int N, int size_per_head, int d_model) 
{ 
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_attn[TILE_SIZE * TILE_SIZE];
    __shared__ float s_v[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        // load attn_scores and V to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_attn[i] = (row < N && col < N) ? attn_scores[head_id * N * N + row * N + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_v[i] = (row < N && col < size_per_head) ? V[row * d_model + head_id * size_per_head + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < size_per_head) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_attn[ty * TILE_SIZE + i] * s_v[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < size_per_head) {
        output[row_c * d_model + col_c + head_id * size_per_head] = acc;
    }
}




// Q, K, V, output are device pointers
void mha_cuda(const float* Q, const float* K, const float* V, float* output, int N,
                      int d_model, int h) 
{
    int size_per_head = d_model / h;
    // qk and scaled
    float* buf;
    int buf_size = h * N * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&buf, buf_size));
    float* qk = buf;
    float* attn_scores = qk + h * N * N;

    // QK^T / sqrt(d_model)
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE), h);
    multi_head_qk_scaled_kernel<<<grid, block>>>(Q, K, qk, N, size_per_head, d_model);

    // Softmax
    softmax_kernel<<<h * N, 256>>>(qk, attn_scores, N);

    // attn_scores * V
    grid = dim3(div_up(size_per_head, TILE_SIZE), div_up(N, TILE_SIZE), h);
    mha_kernel<<<grid, block>>>(attn_scores, V, output, N, size_per_head, d_model);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(buf)); 
}
