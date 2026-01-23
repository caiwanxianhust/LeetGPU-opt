#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cfloat>


static __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void sw_attn_kernel(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_id = blockIdx.x * 4 + warp_id;  // each warp processes one query
    __shared__ float s_qk[4][128];
    for (int i = lane_id; i < 128; i += 32) {
        s_qk[warp_id][i] = -FLT_MAX;
    }
    __syncwarp();

    if (row_id < M) {
        float max_val = -FLT_MAX;
        float sum_exp = 0.0f;

        // compute QK^T for the sliding window
        int k_start_row = (row_id - window_size) >= 0 ? (row_id - window_size) : 0;
        int k_end_row = (row_id + window_size) < M ? (row_id + window_size) : (M - 1);
        int real_window_size = k_end_row - k_start_row + 1;
        float scale = rsqrtf((float)d);
        for (int k_row = k_start_row; k_row <= k_end_row; k_row++) {
            float qk = 0.0f;
            for (int i = lane_id; i < d; i += 32) {
                qk += Q[row_id * d + i] * K[k_row * d + i];
            }
            __syncwarp();
            // warp-level reduction
            qk = warp_all_reduce_sum(qk);
            if (lane_id == 0) {
                qk = qk * scale;
                s_qk[warp_id][k_row - k_start_row] = qk;
                max_val = max(max_val, qk);
            }
        }
        __syncwarp();

        // compute attention output
        if (lane_id == 0) {
            for (int i = 0; i < real_window_size; i++) {
                sum_exp += expf(s_qk[warp_id][i] - max_val);
            }

            sum_exp = 1.0f / sum_exp;

            for (int i = 0; i < real_window_size; i++) {
                // normalize
                s_qk[warp_id][i] = expf(s_qk[warp_id][i] - max_val) * sum_exp;
            }
        }
        __syncwarp();

        for (int i = lane_id; i < d; i += 32) {
            float attn_val = 0.0f;
            for (int j = 0; j < real_window_size; j++) {
                int k_row = k_start_row + j;
                attn_val += s_qk[warp_id][j] * V[k_row * d + i];
            }
            output[row_id * d + i] = attn_val;
        }
    }
}


__global__ void sw_attn_kernel_without_warp_reduce(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) { 
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id < M) {
        float max_val = -FLT_MAX;
        int k_start_row = (row_id - window_size) >= 0 ? (row_id - window_size) : 0;
        int k_end_row = (row_id + window_size) < M ? (row_id + window_size) : (M - 1);
        int real_window_size = k_end_row - k_start_row + 1;

        float scores[65];  // assuming window_size <= 32
        float scale = rsqrtf((float)d);
        // compute max for numerical stability
        for (int k_row = k_start_row; k_row <= k_end_row; k_row++) {
            float qk = 0.0f;
            for (int i = 0; i < d; i++) {
                qk += Q[row_id * d + i] * K[k_row * d + i];
            }
            qk = qk * scale;
            scores[k_row - k_start_row] = qk;
            max_val = max(max_val, qk);
        }

        // compute sum of exp
        float sum_exp = 0.0f;
        for (int i = 0; i < real_window_size; i++) {
            scores[i] = expf(scores[i] - max_val);  // store exp values for later use
            sum_exp += scores[i];
        }

        sum_exp = 1.0f / sum_exp;
        // compute attention output
        for (int i = 0; i < real_window_size; i++) {
            scores[i] *= sum_exp;
        }

        for (int i = 0; i < d; i++) {
            float attn_val = 0.0f;
            for (int j = 0; j < real_window_size; j++) {
                int k_row = k_start_row + j;
                attn_val += scores[j] * V[k_row * d + i];
            }
            output[row_id * d + i] = attn_val;
        }
    }
}


// Q, K, V, output are device pointers
void sliding_window_self_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    // with warp reduce
    // each warp handles one query
    dim3 block(128);
    dim3 grid(div_up(M, 4));
    sw_attn_kernel<<<grid, block>>>(Q, K, V, output, M, d, window_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // without warp reduce
    // each thread handles one query
    // dim3 block(128);
    // dim3 grid(div_up(M, 128));
    // sw_attn_kernel_without_warp_reduce<<<grid, block>>>(Q, K, V, output, M, d, window_size);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

