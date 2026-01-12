#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>


/*
Implement Linear Attention for a given set of matrices, following the method described in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" . Given the query matrix Q of size M×d, key matrix K of size M×d, and value matrix V of size M×d, your program should compute the output matrix using the formula:
 

where 
 is a feature map applied element-wise, for example:
 
 
All matrices Q, K, V, and output are of type float32, and M and d are of type int32.

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

static __device__ __host__ __forceinline__ float phi(float x) {
    return x > 0 ? x + 1.0f : exp(x);
}

static __device__ __host__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static constexpr int TILE_SIZE = 32;


__global__ void zeros_init_kernel(float* vec, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        vec[idx] = 0.0f;
    }
}


/**
 * Compute phi(Q) and phi(K)^T and sum of phi(K)
 * Q: pointer to query matrix of size Mxd
 * K: pointer to key matrix of size Mxd
 * phi_q: pointer to output phi(Q) matrix of size Mxd
 * phi_k_T: pointer to output phi(K)^T matrix of size dxM
 * ksum: pointer to output sum of phi(K) vector of size d
 * M: number of rows
 * d: number of columns
 */
__global__ void phi_qk_ksum_kernel(const float* Q, const float* K, float* phi_q, float* phi_k_T, float* ksum, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / d;
    int col = idx % d;
    if (idx < M * d) {
        phi_q[idx] = phi(Q[idx]);
        phi_k_T[col * M + row] = phi(K[idx]);
        atomicAdd(&ksum[col], phi(K[idx]));
    }
}


/* Compute output = phi(Q) @ sum(phi(K)) */
__global__ void mv_kernel(const float * phi_q, const float* ksum, float* output, int M, int d) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_id = blockIdx.x * 4 + warp_id;
    float sum = 0.0f;

    if (row_id < M) { 
        for (int col = lane_id; col < d; col += 32) {
            sum += phi_q[row_id * d + col] * ksum[col];
        }
        __syncwarp();
        sum = warp_all_reduce_sum(sum);
        if (lane_id == 0) {
            output[row_id] = sum;
        }
    }
}

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) { 
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_a[TILE_SIZE * TILE_SIZE];
    __shared__ float s_b[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load A and B tiles into shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_a = i / TILE_SIZE + blockIdx.y * TILE_SIZE;
            int col_a = i % TILE_SIZE + k;
            s_a[i] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_b = i / TILE_SIZE + k;
            int col_b = i % TILE_SIZE + blockIdx.x * TILE_SIZE;
            s_b[i] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : 0.0f;
        }

        __syncthreads();

        if (row_c < M && col_c < N) {
            for (int n = 0; n < TILE_SIZE; ++n) {
                acc += s_a[ty * TILE_SIZE + n] * s_b[n * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < M && col_c < N) {
        C[row_c * N + col_c] = acc;
    }
}


__global__ void qkv_div_qk_kernel(const float* qkv, const float* phi_qk, float* output, int M, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / d;
    int col = idx % d;
    if (idx < M * d) {
        output[idx] = qkv[idx] / phi_qk[row];
    }
}



// Q, K, V, output are device pointers
void linear_self_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int d) {
    char* d_buf;
    size_t buf_size = d * sizeof(float) + // for ksum
                    M * d * sizeof(float) + // for phi_q
                    M * d * sizeof(float) + // for phi_k_T
                    M * sizeof(float) + // for qk
                    d * d * sizeof(float) + // for kv
                    M * d * sizeof(float); // for qkv
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_buf, buf_size));
    float* ksum = (float*)d_buf;
    float* phi_q = ksum + d;
    float* phi_k_T = phi_q + M * d;
    float* phi_qk = phi_k_T + M * d;
    float* kv = phi_qk + M;
    float* qkv = kv + d * d;

    // Initialize ksum to zero
    zeros_init_kernel<<<div_up(d, 128), 128>>>(ksum, d);

    // Compute phi(Q) and phi(K)^T and sum of phi(K)
    phi_qk_ksum_kernel<<<div_up(M * d, 128), 128>>>(Q, K, phi_q, phi_k_T, ksum, M, d);

    // Compute phi_qk = phi(Q) @ sum(phi(K)) ->  (M x d) @ (d x 1) = (M x 1)
    // each warp computes a row of the output
    mv_kernel<<<div_up(M, 4), 128>>>(phi_q, ksum, phi_qk, M, d);

    // // compute kv = phi(K)^T @ V -> (d x M) @ (M x d) = (d x d)
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(d, TILE_SIZE), div_up(d, TILE_SIZE));
    matmul_kernel<<<grid, block>>>(phi_k_T, V, kv, d, d, M);

    // // compute qkv = phi(Q) @ kv -> (M x d) @ (d x d) = (M x d)
    matmul_kernel<<<dim3(div_up(d, TILE_SIZE), div_up(M, TILE_SIZE)), block>>>(phi_q, kv, qkv, M, d, d);

    // // compute output = qkv / phi_qk -> (M x d) / (M x 1)
    qkv_div_qk_kernel<<<div_up(M * d, 128), 128>>>(qkv, phi_qk, output, M, d);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}
