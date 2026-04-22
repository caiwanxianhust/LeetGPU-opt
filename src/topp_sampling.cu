#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cub/cub.cuh>
#include <cstdint>

#include <vector>
#include <algorithm> // std::sort
#include <numeric>   // std::iota

namespace {

// --- 1. Philox 算法常量定义 (来自 Random123 库标准) ---
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W4x32_0 0x9E3779B9u
#define PHILOX_W4x32_1 0xBB67AE85u

// --- 2. 辅助函数：32位乘法高位截断 ---
__device__ __forceinline__ uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hi) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    *hi = (uint32_t)(product >> 32);
    return (uint32_t)product;
}

// --- 3. Philox 单轮加密 ---
__device__ __forceinline__ void philox4x32_round(uint32_t* ctr, uint32_t* key) {
    uint32_t hi0, hi1;
    uint32_t lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0], &hi0);
    uint32_t lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2], &hi1);
    
    uint32_t t0 = hi1 ^ ctr[1] ^ key[0];
    uint32_t t1 = lo1;
    uint32_t t2 = hi0 ^ ctr[3] ^ key[1];
    uint32_t t3 = lo0;

    ctr[0] = t0; ctr[1] = t1; ctr[2] = t2; ctr[3] = t3;
    
    key[0] += PHILOX_W4x32_0;
    key[1] += PHILOX_W4x32_1;
}

// --- 4. 核心生成器：生成一个 32位随机整数 ---
__device__ __forceinline__ uint32_t philox_gen_uint32(unsigned long long seed, int tid) {
    // 计数器：使用线程ID作为序列号，保证不同线程产生不同随机数
    uint32_t ctr[4] = {0, 0, (uint32_t)tid, (uint32_t)(tid >> 16)}; 
    // 密钥：使用传入的种子
    uint32_t key[2] = {(uint32_t)seed, (uint32_t)(seed >> 32)};

    // 执行 10 轮加密 (标准 Philox 4x32 轮数)
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        philox4x32_round(ctr, key);
    }
    
    // 返回第一个 32位整数
    return ctr[0];
}

// --- 5. 最终接口：生成 [0, threshold_p) 的浮点数 ---
__device__ __forceinline__ float custom_random_uniform(unsigned long long seed, int tid) {
    // 1. 生成随机整数 (范围 0 ~ 2^32-1)
    uint32_t random_int = philox_gen_uint32(seed, tid);
    
    // 2. 转换为 [0, 1) 的浮点数
    // 2^-32 = 2.3283064e-10f
    const float scale = 2.3283064e-10f; 
    float u = (float)random_int * scale;
    return u;
}

} // namespace


namespace cub_sort {

static __global__ void topp_sampling_kernel(
    const float* logits_sorted, const int* word_ids_sorted, int* output, const int* seed,
    const float* p, int N) {
    int seed_state = *seed;
    float threshold_p = *p;
    threshold_p = min(max(threshold_p, 0.0f), 1.0f);
    float max_logit = logits_sorted[0];

    float sum_exp = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        float exp_logit = expf(logits_sorted[i] - max_logit);
        sum_exp += exp_logit;
    }
    __syncthreads();

    sum_exp = block_all_reduce_sum(sum_exp);

    if (tid == 0) {
        // 归一化并计算累积分布用于采样
        float rand_val = custom_random_uniform((unsigned long long)seed_state, threadIdx.x) * threshold_p;

        float current_p = 0.0f;
        int selected_idx = 0;
        for (int i = 0; i < N; i++) {
            current_p += expf(logits_sorted[i] - max_logit) / sum_exp;
            if (current_p >= rand_val) {
                selected_idx = word_ids_sorted[i];
                output[0] = selected_idx; // 将选中的词 id 写回 output[0]
                break;
            }
        }
    }
}

static void launch_topp_sampling_kernel(
    const float* logits_sorted,
    const int* word_ids_sorted,
    int* output,
    const int* seed_state, const float* threshold_p, int N) {
    int block_size = 1024;
    int num_blocks = 1;
    topp_sampling_kernel << <num_blocks, block_size >> > (
        logits_sorted, word_ids_sorted, output, seed_state, threshold_p, N);
}


void topp_sampling_Philox_cuda(const float* logits, const float* p, const int* seed, int* sampled_token, int vocab_size) {
    char* d_buf; // 临时缓冲区指针
    size_t buf_size = vocab_size * sizeof(float) + 2 * vocab_size * sizeof(int); // 计算所需的缓冲区大小
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, buf_size)); // 分配缓冲区
    float* d_logits_sorted = (float*)d_buf; // logits 排序后的结果
    int* d_word_ids = (int*)(d_logits_sorted + vocab_size); // 原始词 id 
    int* d_word_ids_sorted = (int*)(d_word_ids + vocab_size); // 词 id 排序后的结果

    // init word ids
    int* h_word_ids = new int[vocab_size];
    for (int i = 0; i < vocab_size; i++) {
        h_word_ids[i] = i;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_word_ids, h_word_ids, vocab_size * sizeof(int), cudaMemcpyHostToDevice));

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        logits, d_logits_sorted, d_word_ids, d_word_ids_sorted, vocab_size);

    // Allocate temporary storage
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run sorting operation
    cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
        logits, d_logits_sorted, d_word_ids, d_word_ids_sorted, vocab_size);

    // 现在 d_logits_sorted 是排序后的 logits，d_word_ids_sorted 是对应的词 id
    launch_topp_sampling_kernel(d_logits_sorted, d_word_ids_sorted, sampled_token, seed, p, vocab_size);

    delete[] h_word_ids;
    CHECK_CUDA_ERROR(cudaFree(d_buf));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}

} // namespace cub_sort



namespace cpu_sort { 

void cpu_kv_sort_compatible(float* logits, int* word_ids, int vocab_size) {
    // 1. 生成索引数组 [0, 1, 2, ..., vocab_size-1]
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // 2. 对索引数组进行排序
    // 排序规则：比较 logits[索引] 的值，大的排前面（降序）
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    // 3. 根据排好序的索引，重组数据
    // 我们需要临时数组来存放结果，防止覆盖原始数据
    std::vector<float> logits_sorted(vocab_size);
    std::vector<int> ids_sorted(vocab_size);

    for (int i = 0; i < vocab_size; ++i) {
        int original_idx = indices[i]; // 取出第 i 大的元素的原始下标
        logits_sorted[i] = logits[original_idx];
    }

    // 4. 将结果拷回原始数组
    // 如果是在函数内部使用，也可以直接返回 sorted 数组，避免拷回
    std::copy(logits_sorted.begin(), logits_sorted.end(), logits);
    std::copy(indices.begin(), indices.end(), word_ids);
}

void topp_sampling_Philox_cuda(const float* logits, const float* p, const int* seed, int* sampled_token, int vocab_size) {
    char* d_buf; // 临时缓冲区指针
    size_t buf_size = vocab_size * sizeof(float) + vocab_size * sizeof(int); // 计算所需的缓冲区大小
    CHECK_CUDA_ERROR(cudaMalloc(&d_buf, buf_size)); // 分配缓冲区
    float* d_logits_sorted = (float*)d_buf; // logits 排序后的结果
    int* d_word_ids_sorted = (int*)(d_logits_sorted + vocab_size); // 词 id 排序后的结果


    // init word ids
    int* h_word_ids = new int[vocab_size];
    float* h_logits = new float[vocab_size];
    CHECK_CUDA_ERROR(cudaMemcpy(h_logits, logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    cpu_kv_sort_compatible(h_logits, h_word_ids, vocab_size);
    CHECK_CUDA_ERROR(cudaMemcpy(d_logits_sorted, h_logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_word_ids_sorted, h_word_ids, vocab_size * sizeof(int), cudaMemcpyHostToDevice));

    // 现在 d_logits_sorted 是排序后的 logits，d_word_ids_sorted 是对应的词 id
    cub_sort::launch_topp_sampling_kernel(d_logits_sorted, d_word_ids_sorted, sampled_token, seed, p, vocab_size);

    delete[] h_word_ids;
    delete[] h_logits;
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}

} // namespace cpu_sort



void topp_sampling_cuda(const float* logits, const float* p, const int* seed, int* sampled_token, int vocab_size)
{
    // cub_sort::topp_sampling_Philox_cuda(logits, p, seed, sampled_token, vocab_size);
    cpu_sort::topp_sampling_Philox_cuda(logits, p, seed, sampled_token, vocab_size);
}