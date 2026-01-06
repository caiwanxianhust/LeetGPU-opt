#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cfloat>

/*
Implement Attention with Linear Biases (ALiBi), following the method described in "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" , for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix V of size N×d, your program should compute the output matrix using the formula:

where α is a slope controlling the linear bias and Δ = i - j represents the relative position between query i and key j. The softmax function is applied row-wise. Q, K, V, output, and α are all of data type float32; M, N, d are of data type int32.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output matrix output

Constraints
Matrix Q is of size M×d and matrices K and V are of size N×d
1 ≤ M, N ≤ 2048
1 ≤ d ≤ 1024
-1.0 ≤ α ≤ 1.0
*/

namespace {
    constexpr int TILE_SIZE = 32;

    __host__ __device__ __forceinline__ int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
}



/**
 * @brief    QK multiplication, scale, and add delta
 * @param    Q, K, qk are device pointers, Q[M, d], K[N, d], qk[M, N]
 * @param    M, N, d are the dimensions of Q, K
 * @param    alpha is the slope controlling the linear bias
 * @return
 */
__global__ void qk_mul_scale_add_delta(const float* Q, const float* K, float* qk, int M, int N, int d, float alpha)
{ 
    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    const int ty = threadIdx.x / TILE_SIZE;
    const int tx = threadIdx.x % TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;

    float acc = 0.0f;
    for (int k = 0; k < d; k += TILE_SIZE) {
        // load data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_a = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col_a = k + i % TILE_SIZE;
            if (row_a < M && col_a < d) {
                s_a[i] = Q[row_a * d + col_a];
            } else {
                s_a[i] = 0.0f;
            }
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_b = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col_b = k + i % TILE_SIZE;
            if (row_b < N && col_b < d) {
                s_b[i] = K[row_b * d + col_b];
            } else {
                s_b[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_c < M && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_a[ty * TILE_SIZE + i] * s_b[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        qk[row_c * N + col_c] = acc / sqrtf((float)d) + alpha * (row_c - col_c);
    }
}


/**
 * @brief    Softmax kernel
 * @param    qk, attn_score are device pointers, qk[M, N], attn_score[M, N]
 * @param    N is the dimension of K
 * @return
 */
__global__ void softmax(float* qk, float* attn_score, int N)
{
    float max_val = -FLT_MAX;
    float sum = 0.0f;
    float val;

    // find max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        max_val = max(max_val, val);
    }
    __syncthreads();

    max_val = block_all_reduce_max(max_val);

    // find exp sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        sum += expf(val - max_val);
    }
    __syncthreads();
    sum = block_all_reduce_sum(sum);

    // compute softmax
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        attn_score[blockIdx.x * N + i] = expf(val - max_val) / sum;
    }
}

/**
 * @brief    Matrix multiplication kernel
 * @param    A, B, C are device pointers, A[M, K], B[K, N], C[M, N]
 * @param    M, N, K are the dimensions of A, B
 * @return
 */
__global__ void matrix_mul(const float* A, const float* B, float* C, int M, int N, int K)
{
    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    const int ty = threadIdx.x / TILE_SIZE;
    const int tx = threadIdx.x % TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;
    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) { 
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_a = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col_a = k + i % TILE_SIZE;
            if (row_a < M && col_a < K) {
                s_a[i] = A[row_a * K + col_a];
            }
            else {
                s_a[i] = 0.0f;
            }
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_b = k + i / TILE_SIZE;
            int col_b = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            if (row_b < K && col_b < N) {
                s_b[i] = B[row_b * N + col_b];
            }
            else {
                s_b[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_c < M &&  col_c < N) {
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




/**
 * @brief    Attention with Linear Biases
 * @param    Q, K, V, output are device pointers, Q[M, d], K[N, d], V[N, d], output[M, d]
 * @param    M, N, d are the dimensions of Q, K, V
 * @param    alpha is the slope controlling the linear bias
 * @return
 */
// Q, K, V, output are device pointers
void attention_with_linear_bias_cuda(const float* Q, const float* K, const float* V, float* output, int M, int N,
                      int d, float alpha) 
{
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(M, TILE_SIZE));
    float* buf;
    size_t buf_size = M * N * sizeof(float) * 2;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&buf, buf_size));
    float *qk = buf;
    float *attn_score = buf + M * N;
    qk_mul_scale_add_delta<<<grid, block>>>(Q, K, qk, M, N, d, alpha);
    softmax<<<M, 256>>>(qk, attn_score, N);
    matrix_mul<<<grid, block>>>(attn_score, V, output, M, d, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(buf));
}
