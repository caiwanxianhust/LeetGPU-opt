#include "utils.cuh"

#include <assert.h>
#include <torch/extension.h>


/*
Given two matrices a and x, each of shape [B, L] (batch size × sequence length), compute the linear recurrence h of shape [B, L] defined by: h[b, 0] = x[b, 0] and h[b, t] = a[b, t] × h[b, t−1] + x[b, t] for t ≥ 1. All values are float32. This operation is the core computational primitive of State Space Models (SSMs) such as Mamba, S4, and H3.

Linear Recurrence: h[t] = a[t] · h[t-1] + x[t]

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The result must be stored in the output tensor h

Constraints
1 ≤ B ≤ 256 (batch size)
1 ≤ L ≤ 65,536 (sequence length)
All values in a and x are float32
Performance is measured with B = 64, L = 16,384
*/

namespace {
    __host__ __device__ __forceinline__ int div_up(int a, int b) {
        return (a + b - 1) / b;
    }

    constexpr int CHUNK_SIZE = 128;

    /**
     * 32位版本：
     * 输入：无符号整数 x
     * 输出：不小于 x 的最小的 2 的幂 (uint32_t)
     * 范围：适用于 x <= 2^31。如果 x > 2^31，结果会溢出变为 0 (因为 2^32 无法用 32 位表示)。
     */
    uint32_t nextPowerOfTwo_32(unsigned int x) {
        if (x <= 1) {
            return 1;
        }

        // x 已经是无符号，直接操作
        uint32_t n = x;

        // 1. 减 1：处理 x 本身就是 2 的幂的情况
        n--;

        // 2. 位填充：将最高位 1 右侧所有位变为 1
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        // 32位整数只需要移位到 16 即可覆盖所有位

        // 3. 加 1：进位得到 2 的幂
        return n + 1;
    }
}


/**
 * 块内局部扫描 + 提取块总状态
 * grid(num_chunks, B), 其中 num_chunks = div_up(L, CHUNK_SIZE)
 * block(CHUNK_SIZE)
 * h_local: [B, L]   inter block scan result
 * p_local: [B, num_chunks]   每个块的总累积状态 prod，用于后续块间扫描
 * s_local: [B, num_chunks]   每个块的总累积状态 sum，用于后续块间扫描
 * 
 */
__global__ void scan_mul_add_inter_block(
    const float* a, const float* x, float* a_local, float* x_local, float* p_chunk, float* s_chunk, 
    int B, int L, int num_chunks) {
    
    const int batch_id = blockIdx.y;
    const int chunk_id = blockIdx.x;
    if (batch_id >= B || chunk_id >= num_chunks) return;

    const int global_id = batch_id * L + chunk_id * CHUNK_SIZE + threadIdx.x;

    // shared memory for block scan
    __shared__ float s_p[CHUNK_SIZE]; // prefix product
    __shared__ float s_s[CHUNK_SIZE]; // prefix sum

    s_p[threadIdx.x] = (global_id < batch_id * L + L) ? a[global_id] : 1.0f; // out of bound use identity
    s_s[threadIdx.x] = (global_id < batch_id * L + L) ? x[global_id] : 0.0f; // out of bound use identity
    __syncthreads();

    float p_curr, s_curr, p_prev, s_prev;
    // block scan
    for (int offset = 1; offset < CHUNK_SIZE; offset <<= 1) {
        p_curr = s_p[threadIdx.x];
        s_curr = s_s[threadIdx.x];
        if (threadIdx.x >= offset) {
            p_prev = s_p[threadIdx.x - offset];
            s_prev = s_s[threadIdx.x - offset];
        }
        __syncthreads();

        if (threadIdx.x >= offset) {
            s_p[threadIdx.x] = p_curr * p_prev;
            s_s[threadIdx.x] = s_curr + p_curr * s_prev;
        }
        __syncthreads();
    }
    __syncthreads();

    if (global_id < batch_id * L + L) {
        a_local[global_id] = s_p[threadIdx.x];
        x_local[global_id] = s_s[threadIdx.x];
    }

    int last_idx = min(CHUNK_SIZE - 1, L - 1 - chunk_id * CHUNK_SIZE);
    if (threadIdx.x == last_idx && last_idx >= 0) {
        p_chunk[batch_id * num_chunks + chunk_id] = s_p[threadIdx.x];
        s_chunk[batch_id * num_chunks + chunk_id] = s_s[threadIdx.x];
    } 
}


/**
 * 块间扫描 (Block-Level Scan)
 * 每个块负责一个 batch 内的所有 chunk 的扫描结果，输入是每个 chunk 的总状态，输出是每个 chunk 的 prefix 状态
 * grid(B), block(nextPowerOfTwo_32(num_chunks))
 * p_local: [B, num_chunks]   每个块的总累积状态 prod，用于后续块间扫描
 * s_local: [B, num_chunks]   每个块的总累积状态 sum，用于后续块间扫描
 */
__global__ void scan_mul_add_across_block(
    float* p_chunk, float* s_chunk, int B, int num_chunks) { 
    
    const int batch_id = blockIdx.x;

    __shared__ float s_p[1024]; // prefix product
    __shared__ float s_s[1024]; // prefix sum
    s_p[threadIdx.x] = (threadIdx.x < num_chunks) ? p_chunk[batch_id * num_chunks + threadIdx.x] : 1.0f; // out of bound use identity
    s_s[threadIdx.x] = (threadIdx.x < num_chunks) ? s_chunk[batch_id * num_chunks + threadIdx.x] : 0.0f; // out of bound use identity
    __syncthreads();

    float p_curr, s_curr, p_prev, s_prev;
    // block scan
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (threadIdx.x >= offset) {
            p_curr = s_p[threadIdx.x];
            s_curr = s_s[threadIdx.x];
            p_prev = s_p[threadIdx.x - offset];
            s_prev = s_s[threadIdx.x - offset];
        }
        __syncthreads();

        if (threadIdx.x >= offset) {
            s_p[threadIdx.x] = p_curr * p_prev;
            s_s[threadIdx.x] = s_curr + p_curr * s_prev;
        }
        __syncthreads();
    }
    __syncthreads();

    if (threadIdx.x < num_chunks) {
        p_chunk[batch_id * num_chunks + threadIdx.x] = threadIdx.x == 0 ? 1.0f : s_p[threadIdx.x - 1];
        s_chunk[batch_id * num_chunks + threadIdx.x] = threadIdx.x == 0 ? 0.0f : s_s[threadIdx.x - 1];
    }
}


/**
 * 块间扫描 (Block-Level Scan)
 * 每个 thread 负责一个 batch 内的所有 chunk 的扫描结果，输入是每个 chunk 的总状态，输出是每个 chunk 的 prefix 状态
 * grid(divp(B, 128)), block(128)
 * p_local: [B, num_chunks]   每个块的总累积状态 prod，用于后续块间扫描
 * s_local: [B, num_chunks]   每个块的总累积状态 sum，用于后续块间扫描
 */
__global__ void scan_mul_add_across_block_thread(
    float* p_chunk, float* s_chunk, int B, int num_chunks) { 
    
    const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= B) return;

    // 获取当前 batch 的起始索引
    int offset = batch_id * num_chunks;

    float running_p = 1.0f;
    float running_s = 0.0f;

    for (int i = 0; i < num_chunks; ++i) {
        // 1. 读取当前 chunk 的局部状态 (Inclusive state of the chunk itself)
        float local_p = p_chunk[offset + i];
        float local_s = s_chunk[offset + i];

        // 2. 将当前的 running state (即前 i 个 chunk 的累积) 写回，作为第 i 个 chunk 的 prefix (Exclusive)
        p_chunk[offset + i] = running_p;
        s_chunk[offset + i] = running_s;

        // 3. 更新 running state: combine(running, local)
        // new_P = running_P * local_P
        // new_S = running_S * local_P + local_S
        running_s = running_s * local_p + local_s;
        running_p = running_p * local_p;
    }
}


/**
 * 块间扫描 (Warp-Level Scan)
 * 
 * 配置说明:
 * - Grid: div_up(B, 4)
 * - Block: 128 (固定，对应 4 个 Warps)
 * - 每个 Warp (32 threads) 独立负责一个 Batch 的 num_chunks 扫描。
 * - 仅使用寄存器通信 (__shfl_sync)，无共享内存，无 __syncthreads。
 * 
 * 适用场景: num_chunks <= 32。如果 num_chunks > 32，此核函数需要扩展为多级扫描。
 */
__global__ void scan_mul_add_across_block_warp(
    float* p_chunk, float* s_chunk, int B, int num_chunks) { 
    const int warp_id = threadIdx.x / 32;       // 当前 Warp ID (0, 1, 2, 3)
    const int lane_id = threadIdx.x % 32;       // 线程在 Warp 内的 ID (0-31)
    
    // 每个 Block 有 4 个 Warps，所以每个 Block 处理 4 个 Batches
    const int batches_per_block = blockDim.x / 32; 
    const int batch_id = blockIdx.x * batches_per_block + warp_id;

    // 边界检查
    if (batch_id >= B) return;

    // 计算当前 Batch 在全局数组中的起始偏移
    int offset = batch_id * num_chunks;

    // --- 1. 加载数据到寄存器 ---
    // 每个线程加载自己对应的 chunk 数据。
    // 如果 lane_id >= num_chunks，加载单位元 (P=1, S=0)，以符合参考代码逻辑。
    float p_curr = 1.0f;
    float s_curr = 0.0f;

    if (lane_id < num_chunks) {
        p_curr = p_chunk[offset + lane_id];
        s_curr = s_chunk[offset + lane_id];
    }
    __syncwarp(); // 确保所有线程都加载完成，虽然后续操作是基于 lane_id 的条件判断，但为了安全起见加上同步

    // --- 2. Warp 内 Kogge-Stone 扫描 (Inclusive) ---
    // 使用 __shfl_sync 进行寄存器间的数据交换
    // 最大移位距离取决于 num_chunks，但为了代码简单且覆盖 <=32 的情况，循环 5 次 (2^5=32)
    // 内部判断确保只计算有效范围
    for (int shift = 1; shift < 32; shift <<= 1) {
        // 只有当 lane_id >= shift 时，才能从左边拿到有效数据
        // __shfl_sync 会自动处理 lane_id - shift < 0 的情况吗？不会，必须手动保护或依赖逻辑
        // 这里我们利用逻辑：如果 lane_id < shift，我们不更新 p_curr/s_curr，保持原值即可
        
        int src_lane = lane_id - shift;
        // 确保 src_lane 在合法范围 [0, 31] 内，防止 __shfl_sync 未定义行为或线程分歧
        // 如果 src_lane < 0，我们让它读取自己（值不会被使用，因为下面的 if 会过滤）
        int safe_src = (src_lane >= 0) ? src_lane : lane_id;
        
        // 【关键】所有 32 个线程都执行这两行指令，绝对不会有线程跳过
        float p_prev = __shfl_sync(0xffffffff, p_curr, safe_src);
        float s_prev = __shfl_sync(0xffffffff, s_curr, safe_src);
        // 只有当 lane_id >= shift 时，才更新当前线程的状态
        if (lane_id >= shift && lane_id < num_chunks && src_lane < num_chunks) {
            s_curr = s_curr + p_curr * s_prev;
            p_curr = p_curr * p_prev;
        }
        __syncwarp();
    }

    // --- 3. 转换为 Exclusive Scan ---
    // 目前 p_curr/s_curr 是 Inclusive (包含自身)
    // 目标：output[i] = inclusive[i-1], output[0] = identity
    
    float p_out = 1.0f;
    float s_out = 0.0f;

    int src_lane = lane_id - 1;
    int safe_src = (src_lane >= 0) ? src_lane : lane_id;

    // 所有线程执行 shuffle
    float p_inclusive_prev = __shfl_sync(0xffffffff, p_curr, safe_src);
    float s_inclusive_prev = __shfl_sync(0xffffffff, s_curr, safe_src);

    // 只有当 lane_id > 0 时，才使用前一个线程的 Inclusive 值。
    // 当 lane_id == 0 时，必须保持初始化的单位元 (1.0, 0.0)。
    // 之前的代码错误地让 lane_id=0 也使用了 shuffle 的结果（即它自己的值）。
    if (lane_id > 0 && lane_id < num_chunks) {
        p_out = p_inclusive_prev;
        s_out = s_inclusive_prev;
    }
    // else: p_out = 1.0f, s_out = 0.0f (已初始化)

    // --- 4. 写回全局内存 ---
    if (lane_id < num_chunks) {
        p_chunk[offset + lane_id] = p_out;
        s_chunk[offset + lane_id] = s_out;
    }
}



/**
 * 最终结果计算
 * grid(num_chunks, B), block(CHUNK_SIZE)
 * a_local: [B, L]   块内扫描的 prefix product
 * x_local: [B, L]   块内扫描的 prefix sum
 * s_local: [B, num_chunks]   块间扫描的 prefix sum
 * h: [B, L]   输出结果
 * 计算公式：h_final[t] = prefix_S * local_P[t] + local_S[t]
 */
__global__ void scan_mul_add_finalize(
    const float* a_local, const float* x_local,
    const float* s_local, float* h, int L, int num_chunks) {
    
    const int batch_id = blockIdx.y;
    const int chunk_id = blockIdx.x;

    const int global_id = batch_id * L + chunk_id * CHUNK_SIZE + threadIdx.x;

    if (global_id < batch_id * L + L) {
        float local_p = a_local[global_id];
        float local_s = x_local[global_id];
        float prefix_s = s_local[batch_id * num_chunks + chunk_id];

        h[global_id] = local_s + local_p * prefix_s;
    }
}    


__global__ void linear_recurrence_naive_kernel(const float* __restrict__ a,
    const float* __restrict__ x,
    float* __restrict__ h,
    int B, int L)
{
    const int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    const float* a_ptr = a + batch_id * L;
    const float* x_ptr = x + batch_id * L;
    float* h_ptr = h + batch_id * L; 
    if (batch_id < B) {
        h_ptr[0] = x_ptr[0];
        for (int i=1; i<L; ++i) {
            h_ptr[i] = a_ptr[i] * h_ptr[i - 1] + x_ptr[i];
        } 
    }
    
}

// a, x, h are device pointers
void linear_recurrence_naive_cuda(const float* a, const float* x, float* h, int B, int L) {
    const int block_size = 128;
    int grid_size = (B - 1 + block_size) / block_size;
    dim3 block(block_size);
    dim3 grid(grid_size);
    linear_recurrence_naive_kernel<<<grid, block>>>(a, x, h, B, L);
}


// a, x, h are device pointers
void linear_recurrence_scan_cuda(const float* a, const float* x, float* h, int B, int L) 
{
    int num_chunks = div_up(L, CHUNK_SIZE);

    // Allocate intermediate buffers
    float* d_buf;
    size_t buf_size = B * L * sizeof(float) * 2 + B * num_chunks * sizeof(float) * 2; // a_local, x_local, p_chunk, s_chunk
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    float* a_local = d_buf;
    float* x_local = a_local + B * L;
    float* p_chunk = x_local + B * L;
    float* s_chunk = p_chunk + B * num_chunks;

    // Launch kernels
    scan_mul_add_inter_block<<<dim3(num_chunks, B), CHUNK_SIZE>>>(
        a, x, a_local, x_local, p_chunk, s_chunk, B, L, num_chunks);
    CHECK_CUDA_ERROR(cudaGetLastError());

    assert(num_chunks <= 1024); // 确保块间扫描的线程数不超过 1024
    if (num_chunks <= 32) {
        scan_mul_add_across_block_warp<<<div_up(B, 4), 128>>>(
            p_chunk, s_chunk, B, num_chunks);
    }
    else if (num_chunks < 128) {
        scan_mul_add_across_block_thread<<<div_up(B, 128), 128>>>(
            p_chunk, s_chunk, B, num_chunks);
    }
    else {
        scan_mul_add_across_block<<<B, nextPowerOfTwo_32(num_chunks)>>>(
            p_chunk, s_chunk, B, num_chunks);
    }
    CHECK_CUDA_ERROR(cudaGetLastError());

    scan_mul_add_finalize<<<dim3(num_chunks, B), CHUNK_SIZE>>>(
        a_local, x_local, s_chunk, h, L, num_chunks);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}




// a, x, h are device pointers
void linear_recurrence_cuda(const float* a, const float* x, float* h, int B, int L) 
{
    if (L < 128) {
        linear_recurrence_naive_cuda(a, x, h, B, L);
    } else {
        linear_recurrence_scan_cuda(a, x, h, B, L);
    }
}


torch::Tensor linear_recurrence(
    const torch::Tensor& a,
    const torch::Tensor& x
) {
    TORCH_CHECK(a.ndimension() == 2, "a must be 2D");
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(a.size(0) == x.size(0), "The number of rows in a must match the number of rows in x");
    TORCH_CHECK(a.size(1) == x.size(1), "The number of columns in a must match the number of columns in x");
    TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be of type float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be of type float32");

    int B = a.size(0);
    int L = a.size(1);

    torch::Tensor output = torch::empty_like(x);
    linear_recurrence_cuda(
        a.data_ptr<float>(),
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L
    );
    return output;
}