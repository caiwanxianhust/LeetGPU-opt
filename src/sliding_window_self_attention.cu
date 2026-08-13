#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cfloat>
#include <torch/extension.h>




__global__ void sw_attn_warp_reduce_kernel(
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
        for (int k_row = k_start_row; k_row <= k_end_row; k_row++) {
            float qk = 0.0f;
            for (int i = lane_id; i < d; i += 32) {
                qk += Q[row_id * d + i] * K[k_row * d + i];
            }
            __syncwarp();
            // warp-level reduction
            qk = warp_all_reduce_sum(qk);
            if (lane_id == 0) {
                qk = qk / sqrtf((float)d);
                s_qk[warp_id][k_row - k_start_row] = qk;
                max_val = max(max_val, qk);
                // printf("row_id: %d, qk: %f\n", row_id, qk);
            }
        }
        __syncwarp();

        // compute attention output
        if (lane_id == 0) {
            for (int i = 0; i < real_window_size; i++) {
                sum_exp += expf(s_qk[warp_id][i] - max_val);
            }
            // printf("row_id: %d, sum_exp: %f\n", row_id, sum_exp);

            for (int i = 0; i < real_window_size; i++) {
                // normalize
                s_qk[warp_id][i] = expf(s_qk[warp_id][i] - max_val) / sum_exp;
                // printf("row_id: %d, i: %d, s_qk: %f\n", row_id, i, s_qk[warp_id][i]);
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


__global__ void sw_attn_naive_kernel(
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


__global__ void sw_attn_serial_accumulation_kernel(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) { 
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int row_id = blockIdx.x * (blockDim.x >> 5) + warp_id;  // each warp processes one query
    if (row_id >= M) return;

    __shared__ float s_qk[4][128];
    for (int i = lane_id; i < 128; i += 32) {
        s_qk[warp_id][i] = -FLT_MAX;
    }
    __syncwarp();

    float max_val = -FLT_MAX;
    float sum_exp = 0.0f;

    // compute QK^T for the sliding window
    int k_start_row = (row_id - window_size) >= 0 ? (row_id - window_size) : 0;
    int k_end_row = (row_id + window_size) < M ? (row_id + window_size) : (M - 1);
    int real_window_size = k_end_row - k_start_row + 1;
    float scale = rsqrtf((float)d);
    for (int i = lane_id; i < real_window_size; i += 32) {
        int k_row = k_start_row + i;
        float qk = 0.0f;
        for (int j = 0; j < d; j++) {
            qk += Q[row_id * d + j] * K[k_row * d + j];
        }
        qk *= scale;
        s_qk[warp_id][i] = qk;
        max_val = max(max_val, qk);
    }
    __syncwarp();

    max_val = warp_all_reduce_max(max_val);

    for (int i = lane_id; i < real_window_size; i += 32) {
        float val = expf(s_qk[warp_id][i] - max_val);
        s_qk[warp_id][i] = val;
        sum_exp += val;
    }
    __syncwarp();

    sum_exp = warp_all_reduce_sum(sum_exp);

    for (int i = lane_id; i < d; i += 32) {
        float attn_val = 0.0f;
        for (int j = 0; j < real_window_size; j++) {
            int k_row = k_start_row + j;
            attn_val += (s_qk[warp_id][j] / sum_exp) * V[k_row * d + i];
        }
        output[row_id * d + i] = attn_val;
    }
}


void sliding_window_self_attention_warp_reduce_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    // with warp reduce
    // each warp handles one query
    dim3 block(128);
    dim3 grid(div_up(M, 4));
    sw_attn_warp_reduce_kernel<<<grid, block>>>(Q, K, V, output, M, d, window_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void sliding_window_self_attention_naive_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    // each thread handles one query
    dim3 block(128);
    dim3 grid(div_up(M, 128));
    sw_attn_naive_kernel<<<grid, block>>>(Q, K, V, output, M, d, window_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void sliding_window_self_attention_serial_accumulation_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    // without warp reduce, by serial accumulation instead of parallel reduction
    // each warp handles one query
    dim3 block(128);
    dim3 grid(div_up(M, 4));
    sw_attn_serial_accumulation_kernel<<<grid, block>>>(Q, K, V, output, M, d, window_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


// Q, K, V, output are device pointers
void sliding_window_self_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
) {
    sliding_window_self_attention_serial_accumulation_cuda(Q, K, V, output, M, d, window_size);
}


torch::Tensor sliding_window_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int window_size
) { 
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");
    TORCH_CHECK(window_size > 0 && window_size <= 32, "Window size must be positive");

    int M = Q.size(0);
    int d_model = Q.size(1);
    torch::Tensor output = torch::zeros({M, d_model}, Q.options());

    sliding_window_self_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, d_model, window_size
    );
    return output;
}

