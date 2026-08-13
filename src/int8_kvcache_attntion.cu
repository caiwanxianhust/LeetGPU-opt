#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <stdint.h>
#include <torch/extension.h>





namespace flash_attn { 

constexpr int BLOCK_SIZE = 256;
constexpr int Bc = (BLOCK_SIZE >> 5);


/**

 * int8 KV Cache Attention
 * 输入：
    Q: [num_heads, head_dim] (float32)
    K_int8: [num_heads, seq_len, head_dim] (int8)
    V_int8: [num_heads, seq_len, head_dim] (int8)
    k_scale: [num_heads, seq_len] (float32)
    v_scale: [num_heads, seq_len] (float32)
 * 输出：
    output: [num_heads, head_dim] (float32)
 */
__global__ void int8_kvcache_flash_attntion_kernel(
    const float* Q, const int8_t* K_int8, const int8_t* V_int8,
    const float* k_scale, const float* v_scale, float* output, 
    int num_heads, int seq_len, int head_dim, float softmax_scale) 
{
    const int head_id = blockIdx.x;
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const float* Q_ptr = Q + head_id * head_dim;
    const int8_t* K_ptr = K_int8 + head_id * seq_len * head_dim;
    const int8_t* V_ptr = V_int8 + head_id * seq_len * head_dim;
    extern __shared__ float s_buf[];
    float* sQ = s_buf;  // [head_dim]
    float* sK = sQ + head_dim;  // [Bc, head_dim]
    float* sV = sK + Bc * head_dim;     // [Bc, head_dim]
    float* sQK = sV + Bc * head_dim;    // [Bc]
    __shared__ MD_F ml_prev;
    __shared__ MD_F ml_curr;
    __shared__ MD_F ml_new;

    if (threadIdx.x == 0) {
        ml_prev = { -1e20f, 0.0f };;
        ml_curr = { -1e20f, 0.0f };
        ml_new = { -1e20f, 0.0f };
    }
    // load Q from global memory to shared memory
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        sQ[i] = Q_ptr[i];
    }
    __syncthreads();

    // 对 K|V 在 seq_len 维度分组，每组长度为 Bc，共分为 Tc 组
    // 每个 warp 处理一个 K|V 分组
    for (int i = 0; i < seq_len; i+=Bc) {
        int curr_r = i + warp_id;
        // load K|V from global memory to shared memory
        for (int j = lane_id; j < head_dim; j += 32) {
            sK[warp_id * head_dim + j] = (curr_r < seq_len && j < head_dim) ? K_ptr[curr_r * head_dim + j] * k_scale[head_id * seq_len + curr_r] : 0.0f;
            sV[warp_id * head_dim + j] = (curr_r < seq_len && j < head_dim) ? V_ptr[curr_r * head_dim + j] * v_scale[head_id * seq_len + curr_r] : 0.0f;
        }
        __syncthreads();

        // 计算 QK 矩阵，每个 warp 计算一行
        float qk_val = 0.0f;
        for (int j = lane_id; j < head_dim; j += 32) {
            qk_val += sQ[j] * sK[warp_id * head_dim + j];
        }
        __syncwarp();
        qk_val = warp_all_reduce_sum(qk_val);
        if (lane_id == 0) {
            
            sQK[warp_id] = (curr_r < seq_len) ? qk_val * softmax_scale : -1e20f;
        }
        __syncthreads();

        // 计算 softmax
        if (threadIdx.x == 0) {
            MD_F ml_tmp = { -1e20f, 0.0f };
            #pragma unroll
            for (int j = 0; j < Bc; j++) {
                ml_tmp = MDF_Op()(ml_tmp, MD_F({sQK[j], 1.0f}));
            }

            ml_curr = ml_tmp;
            ml_new = MDF_Op()(ml_prev, ml_tmp);
        }
        __syncthreads();

        if (lane_id == 0) {
            sQK[warp_id]  = (curr_r < seq_len) ? __expf(sQK[warp_id] - ml_curr.m) : 0.0f;
        }
        __syncthreads();

        // 计算 QK @ V 矩阵
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
            float qkv_val = 0.0f;
            #pragma unroll
            for (int k = 0; k < Bc; k++) {
                qkv_val += sQK[k] * sV[k * head_dim + j];
            }
            // 直接更新 O 矩阵
            // O[qo_offset + j * d + k] = 1.0f / row_ml_new.d * (row_ml_prev.d * __expf(row_ml_prev.m - row_ml_new.m) * O[qo_offset + j * d + k] + __expf(row_ml.m - row_ml_new.m) * pv);
            output[head_id * head_dim + j] = 1.0f / ml_new.d * (ml_prev.d * __expf(ml_prev.m - ml_new.m) * output[head_id * head_dim + j] + __expf(ml_curr.m - ml_new.m) * qkv_val);
        }

        if (threadIdx.x == 0) {
            ml_prev = ml_new;
        }
        __syncthreads();
    }
}


void launch_int8_kvcache_flash_attntion_kernel(
    const float* Q, const int8_t* K_int8, const int8_t* V_int8,
    const float* k_scale, const float* v_scale, float* output, 
    int num_heads, int seq_len, int head_dim) 
{
    printf("int8_kvcache_attntion_cuda: num_heads=%d, seq_len=%d, head_dim=%d\n", num_heads, seq_len, head_dim);
    float softmax_scale = 1.0f / sqrtf((float)head_dim);
    dim3 block(BLOCK_SIZE);
    dim3 grid(num_heads);
    /*
    float* sQ = s_buf;  // [head_dim]
    float* sK = sQ + head_dim;  // [Bc, head_dim]
    float* sV = sK + Bc * head_dim;     // [Bc, head_dim]
    float* sQK = sV + Bc * head_dim;    // [Bc]
    */
    size_t shared_mem_size = sizeof(float) * (head_dim + Bc * head_dim * 2 + Bc);
    int8_kvcache_flash_attntion_kernel<<<grid, block, shared_mem_size>>>(
        Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim, softmax_scale);
}


}   // namespace flash_attn








/**

 * int8 KV Cache Attention
 * 输入：
    Q: [num_heads, head_dim] (float32)
    K_int8: [num_heads, seq_len, head_dim] (int8)
    V_int8: [num_heads, seq_len, head_dim] (int8)
    k_scale: [num_heads, seq_len] (float32)
    v_scale: [num_heads, seq_len] (float32)
 * 输出：
    output: [num_heads, head_dim] (float32)
 */
// Q, K_int8, V_int8, k_scale, v_scale, output are device pointers
void int8_kvcache_attntion_cuda(
    const float* Q, const int8_t* K_int8, const int8_t* V_int8,
    const float* k_scale, const float* v_scale, float* output, 
    int num_heads, int seq_len, int head_dim) 
{
    flash_attn::launch_int8_kvcache_flash_attntion_kernel(
        Q, K_int8, V_int8, k_scale, v_scale, output, num_heads, seq_len, head_dim);

}


torch::Tensor int8_kvcache_attntion(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& k_scale,
    const torch::Tensor& v_scale
) {
    TORCH_CHECK(Q.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(K.ndimension() == 3, "Input must be 3D");
    TORCH_CHECK(V.ndimension() == 3, "Input must be 3D");
    TORCH_CHECK(k_scale.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(v_scale.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(Q.device().is_cuda() && K.device().is_cuda() && V.device().is_cuda() && k_scale.device().is_cuda() && v_scale.device().is_cuda(), 
        "Input must be on CUDA");
    
    int num_heads = Q.size(0);
    int head_dim = Q.size(1);
    int seq_len = K.size(1);
    
    torch::Tensor output = torch::zeros_like(Q);
    int8_kvcache_attntion_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<int8_t>(),
        V.data_ptr<int8_t>(),
        k_scale.data_ptr<float>(),
        v_scale.data_ptr<float>(),
        output.data_ptr<float>(),
        num_heads, seq_len, head_dim
    );
    return output;
}
