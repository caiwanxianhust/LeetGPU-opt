#include "utils.cuh"
#include <cfloat>

#include "cuda_kernels.cuh"

/*
Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix V of size N×d, your program should compute the output matrix using the formula:
 
where the softmax function is applied row-wise.

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output matrix output

Constraints
Matrix Q is of size M×d and matrices K and V are of size N×d
1 ≤ M, N ≤ 100,000
1 ≤ d ≤ 1024
*/



// flash attention kernel
template <int Bc>
__global__ void softmax_attention_kernel(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{ 
    const int qo_offset = blockIdx.x * d; // Offset for the current query row
    extern __shared__ float s_q[];
    __shared__ float s_qk[Bc];
    __shared__ MD_F s_row_ml_prev;

    // 显式初始化共享变量
    if (threadIdx.x == 0) {
        s_row_ml_prev.m = -FLT_MAX;
        s_row_ml_prev.d = 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        s_q[i] = Q[qo_offset + i];
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        // Initialize s_qk
        for (int i = threadIdx.x; i < Bc; i += blockDim.x) {
            s_qk[i] = 0.0f;
        }
        __syncthreads();

        // Compute dot products
        for (int i = 0; i< Bc && (j + i) < N; ++i) {
            float dot = 0.0f;
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                dot += s_q[k] * K[(j + i) * d + k];
            }
            // Reduce within block
            __syncthreads();
            dot = block_all_reduce_sum(dot);
            
            if (threadIdx.x == 0) {
                s_qk[i] = dot / sqrtf((float)d);
            }
        }
        __syncthreads();

        // Compute softmax and weighted sum
        MD_F row_ml, tmp;
        row_ml.m = -FLT_MAX;
        row_ml.d = 0.0f;
        for (int i=threadIdx.x; i<Bc && (j+i)<N; i+=blockDim.x) {
            tmp.m = s_qk[i];
            tmp.d = 1.0f;
            row_ml = MDF_Op()(row_ml, tmp);
        }
        __syncthreads();
        row_ml = block_all_reduce_sum_mdf(row_ml);
        // Combine with previous blocks
        MD_F row_ml_new = MDF_Op()(s_row_ml_prev, row_ml);

        float rlf = 1.0f / row_ml_new.d;
        float exp_m_diff = expf(s_row_ml_prev.m - row_ml_new.m);
        float exp_m_curr = expf(row_ml.m - row_ml_new.m);

        for (int k = threadIdx.x; k < d; k += blockDim.x) {
            float pv = 0.0f;
            for (int i = 0; i < Bc && (j + i) < N; ++i) {
                pv += expf(s_qk[i] - row_ml.m) * V[(j + i) * d + k];
            }
            
            output[qo_offset + k] = rlf * (exp_m_diff * s_row_ml_prev.d * output[qo_offset + k] + exp_m_curr * pv);
        }

        if (threadIdx.x == 0) {
            s_row_ml_prev = row_ml_new;
        }
        __syncthreads();
    }
}


void softmax_attention_cuda(const float* Q, const float* K, const float* V, float* output, int M, int N, int d)
{
    int block_size = 256;
    int grid_size = M; // One block per output row
    constexpr int Bc = 256;
    int sram_size = sizeof(float) * d;

    softmax_attention_kernel<Bc><<<grid_size, block_size, sram_size, 0>>>(Q, K, V, output, M, N, d);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}