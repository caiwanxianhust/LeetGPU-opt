#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cfloat>


/*
Implement Causal (masked) Self-Attention for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix V of size M×d, your program should compute the output matrix using the formula:

where mask is a causal mask that sets all positions corresponding to keys after the current query to 
.
i.e., for query i and key j:
 
The softmax function is applied row-wise. Q, K, V, and output are all of data type float32; M, and d are of data type int32.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output matrix output

Constraints
Matrix Q, K, and V are all of size M×d
1 ≤ M ≤ 10000
1 ≤ d ≤ 128
All elements in Q, K, and V are sampled from[-100.0, 100.0]
Data type for all matrices is float32
*/

namespace {
    constexpr int TILE_SIZE = 32;
    __host__ __device__ int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
}

/**
 * @brief    QK scaled masked
 * @param    Q, K, qk are device pointers, Q[M, d], K[M, d], qk[M, M]
 * @param    M is the number of rows
 * @param    d is the number of columns
 * @return   void
 */
__global__ void qk_scaled_masked_kernel(const float* Q, const float* K, float* qk, int M, int d) {
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < d; k += TILE_SIZE) {
        // load data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_q[i] = (row < M && col < d) ? Q[row * d + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_k[i] = (row < M && col < d) ? K[row * d + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < M && col_c < M) { 
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < M && col_c < M) {
        qk[row_c * M + col_c] = (row_c >= col_c) ? acc / sqrtf((float)d) : -FLT_MAX;
    }
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[M, M], attn_scores[M, M]
 * @param    rows, cols are the dimensions of qk
 * @return   void
 */
__global__ void softmax_kernel(const float* qk, float* attn_scores, int rows, int cols) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int row_id = blockIdx.x * (blockDim.x / 32) + warp_id;

    float max_val = -FLT_MAX;
    float exp_sum = 0.0f;
    float val;

    if (row_id < rows) { 
        // find max
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = qk[row_id * cols + col_id];
            max_val = max(max_val, val);
        }
        __syncwarp();
        max_val = warp_all_reduce_max(max_val);

        // exp sum
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = exp(qk[row_id * cols + col_id] - max_val);
            exp_sum += val;
        }
        __syncwarp();
        exp_sum = warp_all_reduce_sum(exp_sum);

        // output
        for (int col_id = lane_id; col_id < cols; col_id += 32) {
            val = exp(qk[row_id * cols + col_id] - max_val);
            attn_scores[row_id * cols + col_id] = val / exp_sum;
        }
    }
}


/**
 * @brief    Matrix multiplication
 * @param    A, B, C are device pointers, A[M, K], B[K, N], C[M, N]
 * @param    M, N, K are the dimensions of A, B, C
 * @return   void
 */
__global__ void matmul_fp32_kernel(const float* A, const float* B, float* C, int M, int N, int K) { 
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) { 
        // load data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_a[i] = (row < M && col < K) ? A[row * K + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_b[i] = (row < K && col < N) ? B[row * N + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < M && col_c < N) { 
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_a[ty * TILE_SIZE + i] * s_b[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = acc;
    }
}

// Q, K, V, output are device pointers
void causal_self_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    float* d_buf;
    size_t buf_size = M * M * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, buf_size));
    float* qk = (float*)d_buf;
    float* attn_scores = qk + M * M;

    // qk scaled masked
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(M, TILE_SIZE), div_up(M, TILE_SIZE));
    qk_scaled_masked_kernel<<<grid, block>>>(Q, K, qk, M, d);

    // softmax
    // each warp handles one row
    block = dim3(128);
    grid = dim3(div_up(M, 4));
    softmax_kernel<<<grid, block>>>(qk, attn_scores, M, M);

    // matmul [M, M ] @ [M, d]
    block = dim3(TILE_SIZE * TILE_SIZE);
    grid = dim3(div_up(d, TILE_SIZE), div_up(M, TILE_SIZE));
    matmul_fp32_kernel<<<grid, block>>>(attn_scores, V, output, M, d, M);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}
