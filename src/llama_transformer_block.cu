#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <assert.h>
#include <cfloat>
#include <torch/extension.h>


namespace {
    // parameters for LLAMA
    constexpr int NUM_Q_HEADS = 8;
    constexpr int NUM_KV_HEADS = 2;
    constexpr int D_MODEL = 512;
    constexpr int HEAD_DIM = 64;
    constexpr int FFN_DIM = 1408;
    constexpr float EPS = 1e-5f;


    // Offsets for weights and biases in the weights array
    constexpr int OFF_RMS_SCALE_1 = 0;
    constexpr int OFF_W_Q = 512;
    constexpr int OFF_W_K = 262656;
    constexpr int OFF_W_V = 328192;
    constexpr int OFF_W_O = 393728;
    constexpr int OFF_RMS_SCALE_2 = 655872;
    constexpr int OFF_W_GATE = 656384;
    constexpr int OFF_W_UP = 1377280;
    constexpr int OFF_W_DOWN = 2098176;


    __host__ __device__ inline int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
    
    // 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// 定义宏，用于将 float 数组强制转换为 float2 类型，实现向量化访问
#define FLOAT2(x) (reinterpret_cast<float2 *>(&(x))[0])

// --- 核心配置参数 ---
constexpr int BLOCK_M = 128;      // Block 在 M 维度（行）处理的大小
constexpr int BLOCK_N = 128;      // Block 在 N 维度（列）处理的大小
constexpr int BLOCK_K = 8;        // Block 在 K 维度（深度）处理的大小
constexpr int BLOCK_SIZE = 256;   // 每个 Block 的线程总数

// --- 线程布局配置 (这是该 Kernel 最精妙的部分) ---
// A 矩阵加载时的线程布局：8列 x 32行
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X; // 32

// B 矩阵加载时的线程布局：32列 x 8行
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X; // 8

// B 矩阵（转置）加载时的线程布局：8列 x 32行
constexpr int THREAD_B_T_LAYOUT_X = 8;
constexpr int THREAD_B_T_LAYOUT_Y = BLOCK_SIZE / THREAD_B_T_LAYOUT_X; // 32

// C 矩阵（结果）计算时的线程布局：16列 x 16行
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X; // 16

// 每个线程负责计算的 C 矩阵分块大小
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_X; // 128 / 16 = 8
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y; // 128 / 16 = 8

// Warp 内部细分：在 block 16列 x 16行内将 32 线程的 Warp 分为 8行x4列 的结构，warp 按照 4 x 2 排布
constexpr int THREAD_C_WARP_X = 4;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X; // 8
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X; // 4

}


/**
 * @brief    RMS Layer Normalization
 * @brief    each warp processes one row(hidden_size features) of the input
 * @param    x, output are device pointers, x[seq_len, hidden_size]
 */
static __global__ void rms_norm_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ w, 
    float* __restrict__ output, 
    int seq_len) 
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1f;
    const int row_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
    if (row_id >= seq_len) return;

    float sum = 0.0f;
    #pragma unroll
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        sum += val * val;
    }
    __syncwarp();
    sum = warp_all_reduce_sum(sum) / (float)D_MODEL;

    #pragma unroll
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        output[row_id * D_MODEL + i] = val * rsqrtf(sum + EPS) * w[i];
    }
}

static void launch_rms_norm(const float* x, const float* w, float* output, int seq_len) {
    dim3 block(256);
    dim3 grid(div_up(seq_len, 8));
    rms_norm_kernel<<<grid, block>>>(x, w, output, seq_len);
}


/**
 * @brief    matrix multiplication, tanspose
 * @param    A, B, C are device pointers, A[seq_len, D_MODEL], B[HIDEN_SIZE, D_MODEL], C[NUM_HEADS, seq_len, HEAD_DIM]
 * @return
 */
static __global__ void matmul_rc_pipeline_transpose_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, 
    int M, int N, int K) 
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存分配，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：8列 x 32行
    const int tBx = threadIdx.x & (THREAD_B_T_LAYOUT_X - 1); // 0-7 列索引
    const int tBy = threadIdx.x / THREAD_B_T_LAYOUT_X; // 0-31 行索引
    // C 矩阵计算的线程映射，每个线程计算一个 8x8 的 C 矩阵分块
    int warpId = threadIdx.x >> 5; // 0-7 warpId，32 线程一个 warp
    int laneId = threadIdx.x & 31;
    // C 矩阵计算的 warp 布局 ：2行 x 4列
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-3
    int warpy = warpId / THREAD_C_WARP_DIM_X; // 0-1
    int lanex = laneId & (THREAD_C_WARP_X - 1); // 0-3
    int laney = laneId / THREAD_C_WARP_X; // 0-7
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15 C 矩阵列索引
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15 C 矩阵行索引

    // 4. 寄存器声明
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f}; // 每个线程计算的 C 矩阵分块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 列

    int bufferId = 0; // 双缓冲切换标识

    // 5. 预加载第一块 A 和 B 到共享内存
    // 加载 A 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
            int c = 0 + j + tAx;
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }
    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
        int r = col0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
            int c = 0 + j + tBx;
            sB[0][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的矩阵乘法
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }

            // s2r 预加载数据到寄存器
            if (tk < BLOCK_K) {
                // 加载 A 到寄存器 tCsA
                // 这里使用 float4 向量化加载，一次搬运 4 个 float
                #pragma unroll
                for (int tm = 0; tm < (THREAD_C_Y_TILE_SIZE >> 2); tm++) {
                    int r = ((tCy + tm * THREAD_C_LAYOUT_Y) << 2);  // 行偏移
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }

                #pragma unroll
                for (int tn = 0; tn < (THREAD_C_X_TILE_SIZE >> 2); tn++) {
                    int c = ((tCx + tn * THREAD_C_LAYOUT_X) << 2);  // 列偏移
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c ^ (tk << 2)]);
                }
            }
        }
        // 预加载 下一块 A 和 B 到共享内存
        if (k < K) {
            // 加载 A 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                    int c = k + j + tAx;
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
                int r = col0 + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
                    int c = k + j + tBx;
                    sB[bufferId ^ 1][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        // 切换缓冲区
        bufferId ^= 1;
    }

    // 8. 存储结果
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        // 复杂的位运算重组坐标，确保写回时也是合并访问
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j ++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N) {
                // from [seq_len, num_heads, head_dim] transpose to [num_heads, seq_len, HEAD_DIM]
                int head_id = c / HEAD_DIM;
                int id_in_head = c % HEAD_DIM;
                int dist_id = head_id * M * HEAD_DIM + r * HEAD_DIM + id_in_head;
                C[dist_id] = acc[i][j];
            }
        }
    }
}


static void launch_matmul_rc_pipeline_transpose_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) 
{
    // printf("launch_matmul_rc_pipeline_transpose_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_pipeline_transpose_kernel<<<grid, block>>>(A, B, C, M, N, K);
}


/**
 * @brief rope_embedding
 * @brief each warp process one row
 * @param x: [num_heads, seq_len, D]
 * @param cos: [seq_len, D/2]
 * @param sin: [seq_len, D/2]
 * @param output: [M, D]
 */
static __global__ void rope_embedding_kernel(
    float* __restrict__ x, 
    const float* __restrict__ cos, 
    const float* __restrict__ sin, 
    int num_heads, int seq_len, int D) 
{ 
    const int row_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (row_id >= (num_heads * seq_len)) return;

    float *x_ptr = x + row_id * D;

    float cos_val = cos[(row_id % seq_len) * (D / 2) + threadIdx.x];
    float sin_val = sin[(row_id % seq_len) * (D / 2) + threadIdx.x];
    float q1 = x_ptr[threadIdx.x];
    float q2 = x_ptr[threadIdx.x + 32];

    x_ptr[threadIdx.x] = q1 * cos_val - q2 * sin_val;
    x_ptr[threadIdx.x + 32] = q1 * sin_val + q2 * cos_val;
}

static void launch_rope_embedding_kernel(
    float* x, 
    const float* cos, 
    const float* sin, 
    int num_heads, int seq_len, int D) 
{ 
    assert(D == 64);
    dim3 block(32, 8);
    dim3 grid(div_up(num_heads * seq_len, 8));
    rope_embedding_kernel<<<grid, block>>>(x, cos, sin, num_heads, seq_len, D);
}


/**
 * GQA causal transpose kernel
 * @param Q, K, V, output are device pointers
 * @param Q has shape (num_q_heads, seq_len, head_dim)
 * @param K, V has shape (num_kv_heads, seq_len, head_dim)
 * @param output has shape (seq_len, num_q_heads, head_dim)
 */
static __global__ void gqa_causal_tranaspose_kernel(const float* Q, const float* K, const float* V, float* output,
    int num_q_heads, int num_kv_heads, int seq_len, int head_dim, float scale) { 
    const int q_head_idx = blockIdx.y;
    const int q_seq_idx = blockIdx.x;
    const int kv_head_idx = (q_head_idx * num_kv_heads) / num_q_heads;

    // ptrs to the start of the current query head, key head, and value head
    const float* q_ptr = Q + (q_head_idx * seq_len * head_dim) + (q_seq_idx * head_dim);
    const float* k_ptr = K + (kv_head_idx * seq_len * head_dim);
    const float* v_ptr = V + (kv_head_idx * seq_len * head_dim);
    // output has shape (seq_len, num_q_heads, head_dim)
    float* out_ptr = output + q_seq_idx * num_q_heads * head_dim + q_head_idx * head_dim;
    
    const int warp_id = threadIdx.x >> 5; // 0 to 7
    const int lane_id = threadIdx.x & 31; // 0 to 31

    extern __shared__ float s_ptr[];
    float* s_scores = s_ptr;

    // Q @ K^T * scale
    for (int k_seq_idx = warp_id; k_seq_idx < seq_len; k_seq_idx += (blockDim.x >> 5)) {
        float tmp = 0.0f;
        for (int d = lane_id; d < head_dim; d += 32) {
            tmp += q_ptr[d] * k_ptr[k_seq_idx * head_dim + d];
        }
        __syncwarp();
        // warp reduction to sum tmp across the warp
        tmp = warp_all_reduce_sum(tmp);
        if (lane_id == 0) {
            // causal mask
            s_scores[k_seq_idx] = (k_seq_idx <= q_seq_idx) ? tmp * scale : -FLT_MAX;
        }
    }
    __syncthreads();
    
    // softmax
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        max_val = max(max_val, s_scores[i]);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    float exp_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(s_scores[i] - max_val);
        exp_sum += val;
        s_scores[i] = val;
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    // weights @ V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = 0.0f;
        for (int v_seq_idx = 0; v_seq_idx < seq_len; v_seq_idx++) {
            val += (s_scores[v_seq_idx] / exp_sum) * v_ptr[v_seq_idx * head_dim + d];
        }
        out_ptr[d] = val;
    }
}

// Q, K, V, output are device pointers
// Q has shape (num_q_heads, seq_len, head_dim)
// K, V have shape (num_kv_heads, seq_len, head_dim)
// output has shape (seq_len, num_q_head, head_dim)
static void launch_gqa_causal_transpose_kernel(const float* Q, const float* K, const float* V, float* output,
    int num_q_heads, int num_kv_heads, int seq_len, int head_dim) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    dim3 block(BLOCK_SIZE);
    dim3 grid(seq_len, num_q_heads);
    size_t smem_size = (seq_len) * sizeof(float);
    gqa_causal_tranaspose_kernel<<<grid, block, smem_size>>>(
        Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim, scale);
}



/**
 * @brief    matrix multiplication and residual
 * @param    A, B, C, D are device pointers
 * @brief    D = A * B + C
 * @return
 */
static __global__ void matmul_rc_pipeline_residual_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ C, 
    float* __restrict__ D, 
    int M, int N, int K, bool residual) 
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存分配，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：8列 x 32行
    const int tBx = threadIdx.x & (THREAD_B_T_LAYOUT_X - 1); // 0-7 列索引
    const int tBy = threadIdx.x / THREAD_B_T_LAYOUT_X; // 0-31 行索引
    // C 矩阵计算的线程映射，每个线程计算一个 8x8 的 C 矩阵分块
    int warpId = threadIdx.x >> 5; // 0-7 warpId，32 线程一个 warp
    int laneId = threadIdx.x & 31;
    // C 矩阵计算的 warp 布局 ：2行 x 4列
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // 0-3
    int warpy = warpId / THREAD_C_WARP_DIM_X; // 0-1
    int lanex = laneId & (THREAD_C_WARP_X - 1); // 0-3
    int laney = laneId / THREAD_C_WARP_X; // 0-7
    int tCx = warpx * THREAD_C_WARP_X + lanex; // 0-15 C 矩阵列索引
    int tCy = warpy * THREAD_C_WARP_Y + laney; // 0-15 C 矩阵行索引

    // 4. 寄存器声明
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f}; // 每个线程计算的 C 矩阵分块的累加器，大小为 8x8
    float tCsA[2][THREAD_C_Y_TILE_SIZE]; // A 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 行
    float tCsB[2][THREAD_C_X_TILE_SIZE]; // B 矩阵分块加载到寄存器的临时变量，双缓冲，每个线程负责加载 8 列

    int bufferId = 0; // 双缓冲切换标识

    // 5. 预加载第一块 A 和 B 到共享内存
    // 加载 A 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
            int c = 0 + j + tAx;
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }
    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
        int r = col0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
            int c = 0 + j + tBx;
            sB[0][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的矩阵乘法
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算mma
            if (tk > 0) {
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE; tm++) {
                    #pragma unroll
                    for (int tn = 0; tn < THREAD_C_X_TILE_SIZE; tn++) {
                        acc[tm][tn] += tCsA[(tk - 1) & 1][tm] * tCsB[(tk - 1) & 1][tn];
                    }
                }
            }

            // s2r 预加载数据到寄存器
            if (tk < BLOCK_K) {
                // 加载 A 到寄存器 tCsA
                // 这里使用 float4 向量化加载，一次搬运 4 个 float
                #pragma unroll
                for (int tm = 0; tm < (THREAD_C_Y_TILE_SIZE >> 2); tm++) {
                    int r = ((tCy + tm * THREAD_C_LAYOUT_Y) << 2);  // 行偏移
                    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }

                #pragma unroll
                for (int tn = 0; tn < (THREAD_C_X_TILE_SIZE >> 2); tn++) {
                    int c = ((tCx + tn * THREAD_C_LAYOUT_X) << 2);  // 列偏移
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c ^ (tk << 2)]);
                }
            }
        }
        // 预加载 下一块 A 和 B 到共享内存
        if (k < K) {
            // 加载 A 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                    int c = k + j + tAx;
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
                int r = col0 + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
                    int c = k + j + tBx;
                    sB[bufferId ^ 1][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? B[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        // 切换缓冲区
        bufferId ^= 1;
    }

    // 8. 存储结果
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        // 复杂的位运算重组坐标，确保写回时也是合并访问
        int r = row0 + (tCy << 2) + (i & (~3)) * THREAD_C_LAYOUT_Y + (i & 3);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j ++) {
            int c = col0 + (tCx << 2) + (j & (~3)) * THREAD_C_LAYOUT_X + (j & 3);
            if (r < M && c < N) {
                D[r * N + c] = (residual) ? C[r * N + c] + acc[i][j] : acc[i][j];
            }
        }
    }
}


static void launch_matmul_rc_pipeline_residual_kernel(
    const float* A, const float* B, const float* C, float* D, int M, int N, int K, bool residual) 
{
    // printf("launch_matmul_rc_pipeline_residual_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_rc_pipeline_residual_kernel<<<grid, block>>>(A, B, C, D, M, N, K, residual);
}


__device__ __forceinline__ float silu(float x) 
{
    return x / (1.0f + expf(-1.0f * x));
}

static __global__ void silu_and_pointwise_mul_kernel(
    const float* __restrict__ x_gate,
    const float* __restrict__ x_up,
    float* __restrict__ output,
    int N
) {
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = silu(x_gate[tid]) * x_up[tid];
    }
}


static void launch_silu_and_pointwise_mul_kernel(
    const float* x_gate,
    const float* x_up,
    float* output,
    int N
) {
    const int block_size = 256;
    const int grid_size  = div_up(N, block_size);
    silu_and_pointwise_mul_kernel<<<grid_size, block_size>>>(x_gate, x_up, output, N);
}




/**
 * @brief    LLAMA transformer block
 * @param    x is device pointer, x[seq_len, D_MODEL]
 * @param    output is device pointer, output[seq_len, D_MODEL]
 * @param    weights is device pointer
 * @param    cos, sin are device pointers, cos[seq_len, HEAD_DIM / 2], sin[seq_len, HEAD_DIM / 2]
 */
// x, output, weights, cos, sin are device pointers
void llama_transformer_block_cuda(
    const float* x, float* output, const float* weights, const float* cos,
    const float* sin, int seq_len) 
{
    // Load weights and biases from the weights array
    const float * rms_scale_1 = weights + OFF_RMS_SCALE_1;          // [D_MODEL]
    const float * w_q = weights + OFF_W_Q;                          // [D_MODEL, D_MODEL]   
    const float * w_k = weights + OFF_W_K;                          // [NUM_KV_HEADS * HEAD_DIM, D_MODEL]
    const float * w_v = weights + OFF_W_V;                          // [NUM_KV_HEADS * HEAD_DIM, D_MODEL]   
    const float * w_o = weights + OFF_W_O;                          // [D_MODEL, D_MODEL]   
    const float * rms_scale_2 = weights + OFF_RMS_SCALE_2;          // [D_MODEL]
    const float * w_gate = weights + OFF_W_GATE;                    // [FFN_DIM, D_MODEL]   
    const float * w_up = weights + OFF_W_UP;                        // [FFN_DIM, D_MODEL]
    const float * w_down = weights + OFF_W_DOWN;                    // [D_MODEL, FFN_DIM]   

    // Allocate intermediate buffers
    float* d_buf; // buffer for intermediate results
    float* x_norm_1; // normalized output after first layer norm
    float* Q;
    float* K;
    float* V;
    float* qk;
    float* attn_scores;
    float* attn_out;
    float* attn_projected_res;
    float* x_norm_2;
    float* x_gate;
    float* x_up;
    float* swiglu;

    size_t buf_size = seq_len * D_MODEL * sizeof(float) +       // x_norm_1
                      seq_len * D_MODEL * sizeof(float) * 3 +   // Q K V
                      seq_len * seq_len * sizeof(float) +       // qk
                      seq_len * seq_len * sizeof(float) +       // attn_scores
                      seq_len * D_MODEL * sizeof(float) +       // attn_out
                      seq_len * D_MODEL * sizeof(float) +       // attn_projected_res
                      seq_len * D_MODEL * sizeof(float) +       // x_norm_2
                      seq_len * FFN_DIM * sizeof(float) +       // x_gate
                      seq_len * FFN_DIM * sizeof(float) +       // x_up
                      seq_len * FFN_DIM * sizeof(float);        // swiglu
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    x_norm_1 = d_buf;
    Q = x_norm_1 + seq_len * D_MODEL;
    K = Q + seq_len * D_MODEL;
    V = K + seq_len * D_MODEL;
    qk = V + seq_len * D_MODEL;
    attn_scores = qk + seq_len * seq_len;
    attn_out = attn_scores + seq_len * seq_len;
    attn_projected_res = attn_out + seq_len * D_MODEL;
    x_norm_2 = attn_projected_res + seq_len * D_MODEL;
    x_gate = x_norm_2 + seq_len * D_MODEL;
    x_up = x_gate + seq_len * FFN_DIM;
    swiglu = x_up + seq_len * FFN_DIM;

    // Step 1: RMS Normalization
    // x(seq_len, D_MODEL) -> x_norm_1(seq_len, D_MODEL)
    launch_rms_norm(x, rms_scale_1, x_norm_1, seq_len);

    // Step 2: Multi-Head Attention CAUSAL
    // project q & transpose
    // x_norm_1(seq_len, D_MODEL) @ wq(D_MODEL, D_MODEL) -> q(NUM_q_HEADS, seq_len, HEAD_DIM)
    launch_matmul_rc_pipeline_transpose_kernel(x_norm_1, w_q, Q, seq_len, D_MODEL, D_MODEL);
    // project k & transpose
    // x_norm_1(seq_len, D_MODEL) @ wk(NUM_KV_HEADS * HEAD_DIM, D_MODEL, D_MODEL) -> k(NUM_kv_HEADS, seq_len, HEAD_DIM)
    launch_matmul_rc_pipeline_transpose_kernel(x_norm_1, w_k, K, seq_len, NUM_KV_HEADS * HEAD_DIM, D_MODEL);
    // project v & transpose
    // x_norm_1(seq_len, D_MODEL) @ wv(NUM_KV_HEADS * HEAD_DIM, D_MODEL, D_MODEL) -> v(NUM_kv_HEADS, seq_len, HEAD_DIM)
    launch_matmul_rc_pipeline_transpose_kernel(x_norm_1, w_v, V, seq_len, NUM_KV_HEADS * HEAD_DIM, D_MODEL);
    // rope q k
    launch_rope_embedding_kernel(Q, cos, sin, NUM_Q_HEADS, seq_len, HEAD_DIM);
    launch_rope_embedding_kernel(K, cos, sin, NUM_KV_HEADS, seq_len, HEAD_DIM);
    // qk = q @ k.transpose(-2, -1) / sqrt(head_dim) + Causal_Mask
    // attn_out(seq_len, D_MODEL)
    launch_gqa_causal_transpose_kernel(Q, K, V, attn_out, NUM_Q_HEADS, NUM_KV_HEADS, seq_len, HEAD_DIM);
    // project attn_out & residual
    // attn_out(seq_len, D_MODEL) @ w_o[D_MODEL, D_MODEL] + x(seq_len, D_MODEL)
    launch_matmul_rc_pipeline_residual_kernel(attn_out, w_o, x, attn_projected_res, seq_len, D_MODEL, D_MODEL, true);

    // // Step 3: RMS Normalization
    launch_rms_norm(attn_projected_res, rms_scale_2, x_norm_2, seq_len);

    // Step 4: MLP
    // gate: x_norm_2(seq_len, D_MODEL) @ w_gate[FFN_DIM, D_MODEL] -> x_gate(seq_len, FFN_DIM)
    launch_matmul_rc_pipeline_residual_kernel(x_norm_2, w_gate, nullptr, x_gate, seq_len, FFN_DIM, D_MODEL, false);
    // up: x_norm_2(seq_len, D_MODEL) @ w_up[FFN_DIM, D_MODEL] -> x_up(seq_len, FFN_DIM)
    launch_matmul_rc_pipeline_residual_kernel(x_norm_2, w_up, nullptr, x_up, seq_len, FFN_DIM, D_MODEL, false);
    // silu(x_gate) * x_up
    launch_silu_and_pointwise_mul_kernel(x_gate, x_up, swiglu, seq_len * FFN_DIM);
    // project swiglu & residual
    // swiglu(seq_len, FFN_DIM) @ w_down(D_MODEL, FFN_DIM) + x_norm_2 -> output(seq_len, D_MODEL)
    launch_matmul_rc_pipeline_residual_kernel(swiglu, w_down, attn_projected_res, output, seq_len, D_MODEL, FFN_DIM, true);

    CHECK_CUDA_ERROR(cudaFree(d_buf));

}


torch::Tensor llama_transformer_block(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& cos,
    const torch::Tensor& sin
) {
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(weights.ndimension() == 1, "weights must be 1D");

    int seq_len = x.size(0);
    int d_model = x.size(1);
    torch::Tensor output = torch::empty({seq_len, d_model}, x.options());
    llama_transformer_block_cuda(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        weights.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        seq_len
    );
    return output;
}