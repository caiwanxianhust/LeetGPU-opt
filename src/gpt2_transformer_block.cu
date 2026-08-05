#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>



/*
https://leetgpu.com/challenges/gpt-2-transformer-block
*/


namespace {
    // parameters for GPT-2 small
    constexpr int NUM_HEADS = 12;
    constexpr int D_MODEL = 768;
    constexpr int SIZE_PER_HEAD = D_MODEL / NUM_HEADS;
    constexpr int FFN_DIM = 3072;
    constexpr float EPS = 1e-5f;
    constexpr int TILE_SIZE = 32;

    // Offsets for weights and biases in the weights array
    constexpr int OFF_GAMMA_1 = 0;
    constexpr int OFF_BETA_1 = 768;
    constexpr int OFF_W_QKV = 1536;
    constexpr int OFF_BAIS_QKV = 1771008;
    constexpr int OFF_W_ATTN = 1773312;
    constexpr int OFF_BAIS_ATTN = 2363136;
    constexpr int OFF_GAMMA_2 = 2363904;
    constexpr int OFF_BETA_2 = 2364672;
    constexpr int OFF_W_FC = 2365440;
    constexpr int OFF_BAIS_FC = 4724736;
    constexpr int OFF_W_PROJ = 4727808;
    constexpr int OFF_BAIS_PROJ = 7087104;


    __host__ __device__ inline int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
}





// each warp processes one row(768 features) of the input
__global__ void layer_norm_kernel(const float* x, const float* gamma, const float* beta, float* output, int seq_len) 
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_id = blockIdx.x * 4 + warp_id; // 每个 block 4 个 warp，分别处理 4 行
    if (row_id >= seq_len) return;

    float val_mean = 0.0f;
    float val_var = 0.0f;

    // mean
    for (int i = lane_id; i < D_MODEL; i += 32) {
        val_mean += x[row_id * D_MODEL + i];
    }
    __syncthreads();

    val_mean = warp_all_reduce_sum(val_mean);
    val_mean /= (float)D_MODEL;

    // variance
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        val_var += (val - val_mean) * (val - val_mean);
    }
    __syncthreads();

    val_var = warp_all_reduce_sum(val_var);
    val_var /= (float)D_MODEL;

    // output
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        output[row_id * D_MODEL + i] = (val - val_mean) / sqrtf(val_var + EPS) * gamma[i] + beta[i];
    }
}


void layer_norm(const float* x, const float* gamma, const float* beta, float* output, int seq_len) 
{ 
    int block_size = 128;
    int grid_size = div_up(seq_len, 4); // one warp per row
    layer_norm_kernel<<<grid_size, block_size>>>(x, gamma, beta, output, seq_len);
}


/**
 * 矩阵乘法
 * @param C 输出矩阵，行主序
 * @param A 输入矩阵，行主序
 * @param B 输入矩阵，行主序
 * @param lda 输入矩阵的主维度
 * @param ldb 输入矩阵的主维度
 * @param ldc 输出矩阵的主维度
 */
static __device__ void mma_m16n16k16_fp32(float* C, const float* A, const float* B, int lda, int ldb, int ldc) {
    const int lane_id = threadIdx.x % 32;
    int row_c = lane_id / 8;
    int col_c = lane_id % 8;
    for (int row = row_c; row < 16; row += 4) { 
        for (int col = col_c; col < 16; col += 8) {
            float acc = 0.0f;
            float a_val, b_val;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                a_val = (A[row * lda + k]);
                b_val = (B[k * ldb + col]);
                acc += a_val * b_val;
            }
            C[row * ldc + col] = acc + C[row * ldc + col];
            __syncwarp();
        }
    }
}

template < int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
static __global__ void matmul_bias_warp_kernel(const float* A, const float* B, const float* Bias, float* C, int M, int N, int K) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ float s_a[BM * lda];
    __shared__ float s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    // warp 排布 2x2
    constexpr int warp_col = 2;
    constexpr int warp_row = 2;
    // assert(warp_col * warp_row == blockDim.x / 32);

    // warp 在 BM 和 BN 轴上分别循环 2 次
    constexpr int M_LOOP = BM / (warp_row * MMA_M);
    constexpr int N_LOOP = BN / (warp_col * MMA_N);
    // warp 在 BK 轴上循环 2 次
    constexpr int K_LOOP = BK / (MMA_K);

    // 当前 warp 的索引
    const int warp_col_id = warp_id % (warp_col);
    const int warp_row_id = warp_id / (warp_col);

    // initialize C in shared by zero
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : (0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : (0.0f);
        }
        __syncthreads();

        // compute block mma 64 x 64
        for (int m_id = 0; m_id < M_LOOP; m_id++) {
            for (int n_id = 0; n_id < N_LOOP; n_id++) {
                int a_off = m_id * (MMA_M * warp_row) * lda + warp_row_id * MMA_M * lda;
                int b_off = n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                int c_off = m_id * (MMA_M * warp_row) * ldc + warp_row_id * MMA_M * ldc + n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                for (int k_id = 0; k_id < K_LOOP; k_id++) {
                    a_off += k_id * MMA_K;
                    b_off += k_id * MMA_K * ldb;
                    mma_m16n16k16_fp32(s_c + c_off, s_a + a_off, s_b + b_off, lda, ldb, ldc);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = s_c[row * ldc + col] + Bias[col_c];
        }
    }
}

static void matmul_bias_warp_fp32(const float* A, const float* B, const float* Bias, float* C, int M, int N, int K) { 
    // 每个 warp 计算一个 16x16 的子矩阵，总共 4 个 warp 计算一个 32x32 的子矩阵
    // 每个 block 计算一个 64 x 64 的子矩阵，所以 4 个 warp 分别在 M 和 N 方向上循环 2 次
    // BK = 32, 每个 warp 计算 16x16 矩阵时，需要在 K 维度上计算两次 mma 并累加
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int BK = 32;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int block_size = 128; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_bias_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, Bias, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



namespace {
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


static __device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + tanhf(0.79788456080286535587989211986876f * (x + 0.044715f * x * x * x)));
}


/**
 * 矩阵乘法 C = A * B + bias，使用共享内存进行分块矩阵乘法，并且实现了双缓冲和预加载机制，以隐藏内存访问延迟。
 * @param A, B, C, bias are device pointers, A[M, K], B[K, N], C[M, N], bias[N]
 */
static __global__ void matmul_bias_gelu_res_pipeline_kernel(
    const float* __restrict__ A, const float* __restrict__ B, const float* __restrict__ Bias, 
    const float* __restrict__ front, float* __restrict__ C, 
    int M, int N, int K, bool add_bias, bool add_front, bool use_gelu) 
{ 
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 分配共享内存，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：32列 x 8行
    const int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-31 列索引
    const int tBy = threadIdx.x / THREAD_B_LAYOUT_X; // 0-7 行索引
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
            sA[0][j + tAx][(i + tAy) ^ (tAx << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }

    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
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
                // 加载 A 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ (tk << 2)]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个float
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4)]);
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
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }

    // 7. 将计算结果写回全局内存
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                float front_val = (add_front) ? front[r * N + c] : 0.0f;
                float bias_val = (add_bias) ? Bias[c] : 0.0f;
                float res = acc[i][j] + bias_val + front_val;
                C[r * N + c] = (use_gelu) ? gelu(res) : res;
            }
        }
    }
}


static void launch_matmul_bias_gelu_res_pipeline_kernel(
    const float* A, const float* B, const float* Bias, const float* front, float* C, 
    int M, int N, int K, bool add_bias, bool add_front, bool use_gelu) {
    // printf("launch_matmul_bias_gelu_res_pipeline_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    matmul_bias_gelu_res_pipeline_kernel<<<grid, block>>>(A, B, Bias, front, C, M, N, K, add_bias, add_front, use_gelu);
}




template < int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
__global__ void matmul_bias_res_warp_kernel(const float* A, const float* B, const float* Bias, const float* front, float* C, int M, int N, int K) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ float s_a[BM * lda];
    __shared__ float s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    // warp 排布 2x2
    constexpr int warp_col = 2;
    constexpr int warp_row = 2;
    // assert(warp_col * warp_row == blockDim.x / 32);

    // warp 在 BM 和 BN 轴上分别循环 2 次
    constexpr int M_LOOP = BM / (warp_row * MMA_M);
    constexpr int N_LOOP = BN / (warp_col * MMA_N);
    // warp 在 BK 轴上循环 2 次
    constexpr int K_LOOP = BK / (MMA_K);

    // 当前 warp 的索引
    const int warp_col_id = warp_id % (warp_col);
    const int warp_row_id = warp_id / (warp_col);

    // initialize C in shared by zero
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : (0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : (0.0f);
        }
        __syncthreads();

        // compute block mma 64 x 64
        for (int m_id = 0; m_id < M_LOOP; m_id++) {
            for (int n_id = 0; n_id < N_LOOP; n_id++) {
                int a_off = m_id * (MMA_M * warp_row) * lda + warp_row_id * MMA_M * lda;
                int b_off = n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                int c_off = m_id * (MMA_M * warp_row) * ldc + warp_row_id * MMA_M * ldc + n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                for (int k_id = 0; k_id < K_LOOP; k_id++) {
                    a_off += k_id * MMA_K;
                    b_off += k_id * MMA_K * ldb;
                    mma_m16n16k16_fp32(s_c + c_off, s_a + a_off, s_b + b_off, lda, ldb, ldc);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = s_c[row * ldc + col] + Bias[col_c] + front[row_c * N + col_c];
        }
    }
}

void matmul_bias_res_warp_fp32(const float* A, const float* B, const float* Bias, const float* front, float* C, int M, int N, int K) { 
    // 每个 warp 计算一个 16x16 的子矩阵，总共 4 个 warp 计算一个 32x32 的子矩阵
    // 每个 block 计算一个 64 x 64 的子矩阵，所以 4 个 warp 分别在 M 和 N 方向上循环 2 次
    // BK = 32, 每个 warp 计算 16x16 矩阵时，需要在 K 维度上计算两次 mma 并累加
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int BK = 32;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int block_size = 128; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_bias_res_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, Bias, front, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



/**
 * @brief    Transpose and split QKV
 * @param    qkv, Q, K, V are device pointers, qkv[seq_len, 3 * dmodel], 
 * @return Q[num_heads, seq_len, size_per_head], K[num_heads, seq_len, size_per_head], V[num_heads, seq_len, size_per_head]
 */
static __global__ void transpose_split_qkv_kernel(
    const float* __restrict__ qkv, float* __restrict__ Q, float* __restrict__ K, float* __restrict__ V, 
    int seq_len, int num_heads, int size_per_head)
{
    const int qkv_idx = blockIdx.z; // 0: Q, 1: K, 2: V
    const int head_id = blockIdx.y;
    const int seq_id = blockIdx.x;
    const int feature_id = threadIdx.x;
    float* dist_ptr = (qkv_idx == 0) ? Q : ((qkv_idx == 1) ? K : V);

    const int src_idx = seq_id * (3 * num_heads * size_per_head) + qkv_idx * num_heads * size_per_head + head_id * size_per_head + feature_id;
    const int dist_idx = head_id * (seq_len * size_per_head) + seq_id * size_per_head + feature_id;
    dist_ptr[dist_idx] = qkv[src_idx];
}

static void launch_transpose_split_qkv_kernel(
    const float* qkv, float* Q, float* K, float* V, int seq_len, int num_heads, int size_per_head) 
{
    dim3 grid(seq_len, num_heads, 3);
    dim3 block(size_per_head);
    transpose_split_qkv_kernel<<<grid, block>>>(qkv, Q, K, V, seq_len, num_heads, size_per_head);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


/**
 * @brief    QK multiplication, scale, and add delta
 * @param    A, B, C are device pointers, A[batch, M, d], B[batch, N, d], C[batch, M, N]
 * @param    scale is the scaling factor for the dot product, typically 1/sqrt(d)
 * @return
 */
static __global__ void batched_matmul_rc_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale, int batch_size) 
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const float *a_ptr = A + blockIdx.z * M * K;
    const float *b_ptr = B + blockIdx.z * N * K;
    float *c_ptr = C + blockIdx.z * M * N;

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
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
        }
    }
    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
        int r = col0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
            int c = 0 + j + tBx;
            sB[0][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? b_ptr[r * K + c] : 0.0f;
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
                // 加载 A 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ ((tk << 2))]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ ((tk << 2))]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c ^ ((tk << 2))]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4) ^ ((tk << 2))]);
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
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? a_ptr[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += THREAD_B_T_LAYOUT_Y) {
                int r = col0 + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_B_T_LAYOUT_X) {
                    int c = k + j + tBx;
                    sB[bufferId ^ 1][j + tBx][(i + tBy) ^ ((j + tBx) << 2)] = (r < N && c < K) ? b_ptr[r * K + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        // 切换缓冲区
        bufferId ^= 1;
    }

    // 7. 将计算结果写回全局内存，写回时进行 scale 和 bias 的处理
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r =  row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                c_ptr[r * N + c] = acc[i][j] * scale;
            }
        }
    }
}


static void launch_batched_matmul_rc_pipeline_scale_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, float scale, int batch_size) 
{
    // printf("launch_batched_matmul_rc_pipeline_scale_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M), batch_size);
    batched_matmul_rc_pipeline_scale_kernel<<<grid, block>>>(A, B, C, M, N, K, scale, batch_size);
}


/**
 * 批量矩阵乘法+转置
 * @param A, B, C are device pointers, A[batch, M, d], B[batch, N, d], C[M, batch, N]
 */
static __global__ void batched_matmul_pipeline_transpose_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, int batch_size)
{
    // 1. 当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;
    const float* A_ptr = A + blockIdx.z * M * K;
    const float* B_ptr = B + blockIdx.z * N * K;

    // 2. 分配共享内存，双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // A 的分块，大小为 BLOCK_K x BLOCK_M，双缓冲
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // B 的分块，大小为 BLOCK_K x BLOCK_N，双缓冲

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：32列 x 8行
    const int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1); // 0-31 列索引
    const int tBy = threadIdx.x / THREAD_B_LAYOUT_X; // 0-7 行索引
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
            sA[0][j + tAx][(i + tAy) ^ (tAx << 2)] = (r < M && c < K) ? A_ptr[r * K + c] : 0.0f;
        }
    }

    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
        int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B_ptr[r * N + c] : 0.0f;
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
                // 加载 A 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ (tk << 2)]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个float
                #pragma unroll
                for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 3; tn++) {
                    int c = (tn * THREAD_C_LAYOUT_X + tCx) << 3;
                    FLOAT4(tCsB[tk & 1][tn << 3]) = FLOAT4(sB[bufferId][tk][c]);
                    FLOAT4(tCsB[tk & 1][(tn << 3) + 4]) = FLOAT4(sB[bufferId][tk][(c + 4)]);
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
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A_ptr[r * K + c] : 0.0f;
                }
            }

            // 加载 B 的下一块
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B_ptr[r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }

        bufferId ^= 1; // 切换缓冲区
    }

    // 7. 将计算结果写回全局内存
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r = row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                // 转置后的索引
                int dist_idx = r * batch_size * N + blockIdx.z * N + c;
                C[dist_idx] = acc[i][j];
            }
        }
    }
}


static void launch_batched_matmul_pipeline_transpose_kernel(
    const float* A, const float* B, float* C, int M, int N, int K, int batch_size) 
{
    // printf("launch_batched_matmul_pipeline_transpose_kernel with M=%d, N=%d, K=%d\n", M, N, K);
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M), batch_size);
    batched_matmul_pipeline_transpose_kernel<<<grid, block>>>(A, B, C, M, N, K, batch_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



/**
 * @brief    Multi-head QK scaled
 * @param    Q, K, qk are device pointers, Q[N, h, size_per_head], K[N, h, size_per_head], qk[h, N, N]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
static __global__ void qk_scaled_kernel(const float* Q, const float*K, float* qk, int N, int size_per_head, int d_model) 
{
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int offset_c = head_id * N * N;
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < size_per_head; k += TILE_SIZE) {
        // load Q and K to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_q[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? Q[row * d_model + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_k[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? K[row * d_model + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < N) {
        qk[offset_c + row_c * N + col_c] = acc / sqrtf((float)size_per_head);
    }
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[h, N, N], attn_scores[h, N, N]
 * @param    N sequence length
 * @return   void
 */
static __global__ void softmax_kernel(const float* qk, float* attn_scores, int N) 
{
    float max_val = -INFINITY;
    float exp_sum = 0.0f;
    float val;

    // Find max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    // Compute exp sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        exp_sum += expf(val - max_val);
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    // Write output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        attn_scores[blockIdx.x * N + i] = expf(val - max_val) / exp_sum;
    }
}


/**
 * @brief    Multi-head attention
 * @param    attn_scores, V, output are device pointers, attn_scores[h, N, N], V[N, h, size_per_head], output[N, h, size_per_head]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void multi_head_attention_kernel(const float* attn_scores, const float* V, float* output, int N, int size_per_head, int d_model) 
{ 
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_attn[TILE_SIZE * TILE_SIZE];
    __shared__ float s_v[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        // load attn_scores and V to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_attn[i] = (row < N && col < N) ? attn_scores[head_id * N * N + row * N + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_v[i] = (row < N && col < size_per_head) ? V[row * d_model + head_id * size_per_head + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < size_per_head) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_attn[ty * TILE_SIZE + i] * s_v[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < size_per_head) {
        output[row_c * d_model + col_c + head_id * size_per_head] = acc;
    }
}




// Q, K, V, output are device pointers
void multi_head_attention(const float* Q, const float* K, const float* V, float* qk, float* attn_scores, float* attn_out, int N) 
{
    
    // QK^T / sqrt(d_model) -> qk[h, N, N]
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE), NUM_HEADS);
    qk_scaled_kernel<<<grid, block>>>(Q, K, qk, N, SIZE_PER_HEAD, D_MODEL);
    // float scale = 1.0f / sqrtf((float)SIZE_PER_HEAD);
    // launch_batched_matmul_rc_pipeline_scale_kernel(Q, K, qk, N, N, SIZE_PER_HEAD, scale, NUM_HEADS);

    // Softmax -> attn_scores[h, N, N]
    softmax_kernel<<<NUM_HEADS * N, 256>>>(qk, attn_scores, N);

    // attn_scores * V -> attn_out[N, h, SIZE_PER_HEAD]
    grid = dim3(div_up(SIZE_PER_HEAD, TILE_SIZE), div_up(N, TILE_SIZE), NUM_HEADS);
    multi_head_attention_kernel<<<grid, block>>>(attn_scores, V, attn_out, N, SIZE_PER_HEAD, D_MODEL);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // attn_scores[h, N, N] @ V[h, N, SIZE_PER_HEAD] -> [h, N, SIZE_PER_HEAD] -> attn_out[N, h, SIZE_PER_HEAD]
    // launch_batched_matmul_pipeline_transpose_kernel(attn_scores, V, attn_out, N, SIZE_PER_HEAD, N, NUM_HEADS);
}


/**
 * @brief    Multi-head attention wrapperq  
 * @param    Q, K, V, qk, attn_scores, attn_out are device pointers, 
 * @param    Q[h, N, size_per_head], K[h, N, size_per_head], V[h, N, size_per_head], qk[h, N, N], attn_scores[h, N, N], attn_out[N, h, size_per_head]
 * @param    N sequence length
 */
static void mha(const float* Q, const float* K, const float* V, float* qk, float* attn_scores, float* attn_out, int N) {
    float scale = 1.0f / sqrtf((float)SIZE_PER_HEAD);
    launch_batched_matmul_rc_pipeline_scale_kernel(Q, K, qk, N, N, SIZE_PER_HEAD, scale, NUM_HEADS);

    // Softmax -> attn_scores[h, N, N]
    softmax_kernel<<<NUM_HEADS * N, 256>>>(qk, attn_scores, N);

    // attn_scores[h, N, N] @ V[h, N, SIZE_PER_HEAD] -> [h, N, SIZE_PER_HEAD] -> attn_out[N, h, SIZE_PER_HEAD]
    launch_batched_matmul_pipeline_transpose_kernel(attn_scores, V, attn_out, N, SIZE_PER_HEAD, N, NUM_HEADS);
}

static __global__ void gelu_kernel(const float* input, float* output, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

static void launch_gelu(const float* input, float* output, int size) 
{
    int block_size = 256;
    int grid_size = div_up(size, block_size);
    gelu_kernel<<<grid_size, block_size>>>(input, output, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



// x, output, weights are device pointers
void gpt2_transformer_block_naive_cuda(const float* x, float* output, const float* weights, int seq_len) 
{
    // Load weights and biases from the weights array
    const float* gamma_1 = weights + OFF_GAMMA_1;
    const float* beta_1 = weights + OFF_BETA_1;
    const float* w_qkv = weights + OFF_W_QKV;
    const float* bias_qkv = weights + OFF_BAIS_QKV;
    const float* w_attn = weights + OFF_W_ATTN;
    const float* bias_attn = weights + OFF_BAIS_ATTN;
    const float* gamma_2 = weights + OFF_GAMMA_2;
    const float* beta_2 = weights + OFF_BETA_2;
    const float* w_fc = weights + OFF_W_FC;
    const float* bias_fc = weights + OFF_BAIS_FC;
    const float* w_proj = weights + OFF_W_PROJ;
    const float* bias_proj = weights + OFF_BAIS_PROJ;

    // Allocate intermediate buffers
    float* d_buf; // buffer for intermediate results
    float* x_norm_1; // normalized output after first layer norm
    float* qkv; // concatenated Q, K, V
    float* Q;
    float* K;
    float* V;
    float* qk;
    float* attn_scores;
    float* attn_out;
    float* attn_projected_res;
    float* x_norm_2;
    float* fc_linear;
    float* fc_gelu;

    int buf_size = seq_len * D_MODEL * sizeof(float) +                                      // x_norm_1
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // qkv
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // Q, K, V
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // qk       
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // attn_scores
    seq_len * D_MODEL * sizeof(float) +                                                      // attn_out
    seq_len * D_MODEL * sizeof(float) +                                                       // attn_projected
    seq_len * D_MODEL * sizeof(float) +                                                     // x_norm_2
    seq_len * FFN_DIM * sizeof(float) +                                                           // fc_linear
    seq_len * FFN_DIM * sizeof(float);                                                            // fc_gelu

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    x_norm_1 = d_buf;
    qkv = x_norm_1 + seq_len * D_MODEL;
    Q = qkv + seq_len * 3 * D_MODEL;
    K = Q + seq_len * D_MODEL;
    V = K + seq_len * D_MODEL;
    qk = V + seq_len * D_MODEL;
    attn_scores = qk + NUM_HEADS * seq_len * seq_len;
    attn_out = attn_scores + NUM_HEADS * seq_len * seq_len;
    attn_projected_res = attn_out + seq_len * D_MODEL;
    x_norm_2 = attn_projected_res + seq_len * D_MODEL;
    fc_linear = x_norm_2 + seq_len * D_MODEL;
    fc_gelu = fc_linear + seq_len * FFN_DIM;


    // Step 1: Layer Normalization
    // x(N, D_MODEL) -> x_norm(N, D_MODEL)
    layer_norm(x, gamma_1, beta_1, x_norm_1, seq_len);

    // Step 2: matmul and add bias for QKV
    // x_norm(N, D_MODEL) * w_qkv(D_MODEL, 3 * D_MODEL) + bias_qkv(3 * D_MODEL) -> qkv(N, 3 * D_MODEL)
    matmul_bias_warp_fp32(x_norm_1, w_qkv, bias_qkv, qkv, seq_len, 3 * D_MODEL, D_MODEL);
    
    // step 3: split QKV
    // qkv(N, 3 * D_MODEL) -> Q(N, D_MODEL), K(N, D_MODEL), V(N, D_MODEL)
    CHECK_CUDA_ERROR(cudaMemcpy2D(Q, D_MODEL * sizeof(float), qkv,               3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy2D(K, D_MODEL * sizeof(float), qkv + D_MODEL,     3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy2D(V, D_MODEL * sizeof(float), qkv + 2 * D_MODEL, 3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));
    
    // step 4: compute attention
    // QK^T / sqrt(d_model) -> attn_scores(N, N)
    // attn_scores(N, N) * V(N, D_MODEL) -> attn_out(N, D_MODEL)
    multi_head_attention(Q, K, V, qk, attn_scores, attn_out, seq_len);
    
    // attn projection
    // attn_out(N, D_MODEL) * w_attn(D_MODEL, D_MODEL) + bias_attn(D_MODEL) + x(N, D_MODEL) -> attn_projected(N, D_MODEL)
    matmul_bias_res_warp_fp32(attn_out, w_attn, bias_attn, x, attn_projected_res, seq_len, D_MODEL, D_MODEL);

    // step 5: Layer Normalization
    layer_norm(attn_projected_res, gamma_2, beta_2, x_norm_2, seq_len);

    // step 6: FFN
    // x_norm_2(N, D_MODEL) * w_fc(D_MODEL, FFN_DIM) + bias_fc(FFN_DIM) -> fc_linear(N, FFN_DIM)
    matmul_bias_warp_fp32(x_norm_2, w_fc, bias_fc, fc_linear, seq_len, FFN_DIM, D_MODEL);
    
    // step 7: GeLU
    // fc_linear(N, FFN_DIM) -> fc_gelu(N, FFN_DIM)
    launch_gelu(fc_linear, fc_gelu, seq_len * FFN_DIM);

    // step 8: FFN projection
    // fc_gelu(N, FFN_DIM) * w_proj(FFN_DIM, D_MODEL) + bias_proj(D_MODEL) + attn_projected(N, D_MODEL) -> output(N, D_MODEL)
    matmul_bias_res_warp_fp32(fc_gelu, w_proj, bias_proj, attn_projected_res, output, seq_len, D_MODEL, FFN_DIM);
   
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


// x, output, weights are device pointers
void gpt2_transformer_block_pipeline_matmul_cuda(const float* x, float* output, const float* weights, int seq_len) 
{
    // Load weights and biases from the weights array
    const float* gamma_1 = weights + OFF_GAMMA_1;
    const float* beta_1 = weights + OFF_BETA_1;
    const float* w_qkv = weights + OFF_W_QKV;
    const float* bias_qkv = weights + OFF_BAIS_QKV;
    const float* w_attn = weights + OFF_W_ATTN;
    const float* bias_attn = weights + OFF_BAIS_ATTN;
    const float* gamma_2 = weights + OFF_GAMMA_2;
    const float* beta_2 = weights + OFF_BETA_2;
    const float* w_fc = weights + OFF_W_FC;
    const float* bias_fc = weights + OFF_BAIS_FC;
    const float* w_proj = weights + OFF_W_PROJ;
    const float* bias_proj = weights + OFF_BAIS_PROJ;

    // Allocate intermediate buffers
    float* d_buf; // buffer for intermediate results
    float* x_norm_1; // normalized output after first layer norm
    float* qkv; // concatenated Q, K, V
    float* Q;
    float* K;
    float* V;
    float* qk;
    float* attn_scores;
    float* attn_out;
    float* attn_projected_res;
    float* x_norm_2;
    float* fc_gelu;

    int buf_size = seq_len * D_MODEL * sizeof(float) +                                      // x_norm_1
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // qkv
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // Q, K, V
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // qk       
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // attn_scores
    seq_len * D_MODEL * sizeof(float) +                                                      // attn_out
    seq_len * D_MODEL * sizeof(float) +                                                       // attn_projected
    seq_len * D_MODEL * sizeof(float) +                                                     // x_norm_2
    seq_len * FFN_DIM * sizeof(float);                                                            // fc_gelu

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    x_norm_1 = d_buf;
    qkv = x_norm_1 + seq_len * D_MODEL;
    Q = qkv + seq_len * 3 * D_MODEL;
    K = Q + seq_len * D_MODEL;
    V = K + seq_len * D_MODEL;
    qk = V + seq_len * D_MODEL;
    attn_scores = qk + NUM_HEADS * seq_len * seq_len;
    attn_out = attn_scores + NUM_HEADS * seq_len * seq_len;
    attn_projected_res = attn_out + seq_len * D_MODEL;
    x_norm_2 = attn_projected_res + seq_len * D_MODEL;
    fc_gelu = x_norm_2 + seq_len * D_MODEL;


    // Step 1: Layer Normalization
    // x(N, D_MODEL) -> x_norm(N, D_MODEL)
    layer_norm(x, gamma_1, beta_1, x_norm_1, seq_len);

    // Step 2: matmul and add bias for QKV
    // x_norm(N, D_MODEL) * w_qkv(D_MODEL, 3 * D_MODEL) + bias_qkv(3 * D_MODEL) -> qkv(N, 3 * D_MODEL)
    launch_matmul_bias_gelu_res_pipeline_kernel(x_norm_1, w_qkv, bias_qkv, nullptr, qkv, seq_len, 3 * D_MODEL, D_MODEL, true, false, false);

    // step 3: split QKV
    // qkv(N, 3 * D_MODEL) -> Q(NUM_HEADS, N, size_per_head), K(NUM_HEADS, N, size_per_head), V(NUM_HEADS, N, size_per_head)
    launch_transpose_split_qkv_kernel(qkv, Q, K, V, seq_len, NUM_HEADS, SIZE_PER_HEAD);

    // step 4: compute attention
    // QK^T / sqrt(d_model) -> attn_scores(NUM_HEADS, N, N)
    // attn_scores(NUM_HEADS, N, N) * V(NUM_HEADS, N, size_per_head) -> attn_out(NUM_HEADS, N, size_per_head) -> attn_out(N, D_MODEL)
    mha(Q, K, V, qk, attn_scores, attn_out, seq_len);
    // attn projection
    // attn_out(N, D_MODEL) * w_attn(D_MODEL, D_MODEL) + bias_attn(D_MODEL) + x(N, D_MODEL) -> attn_projected(N, D_MODEL)
    launch_matmul_bias_gelu_res_pipeline_kernel(attn_out, w_attn, bias_attn, x, attn_projected_res, 
        seq_len, D_MODEL, D_MODEL, true, true, false);


    // step 5: Layer Normalization
    layer_norm(attn_projected_res, gamma_2, beta_2, x_norm_2, seq_len);

    // step 6: FFN + GELU fusion
    // x_norm_2(N, D_MODEL) * w_fc(D_MODEL, FFN_DIM) + bias_fc(FFN_DIM) -> fc_gelu(N, FFN_DIM)
    launch_matmul_bias_gelu_res_pipeline_kernel(x_norm_2, w_fc, bias_fc, nullptr, fc_gelu, 
        seq_len, FFN_DIM, D_MODEL, true, false, true);

    // step 7: FFN projection
    // fc_gelu(N, FFN_DIM) * w_proj(FFN_DIM, D_MODEL) + bias_proj(D_MODEL) + attn_projected(N, D_MODEL) -> output(N, D_MODEL)
    launch_matmul_bias_gelu_res_pipeline_kernel(fc_gelu, w_proj, bias_proj, attn_projected_res, output, 
        seq_len, D_MODEL, FFN_DIM, true, true, false);

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


void gpt2_transformer_block_cuda(const float* x, float* output, const float* weights, int seq_len) 
{
    // gpt2_transformer_block_naive_cuda(x, output, weights, seq_len);
    gpt2_transformer_block_pipeline_matmul_cuda(x, output, weights, seq_len);
}


torch::Tensor gpt2_transformer_block(
    const torch::Tensor& x,
    const torch::Tensor& weights
) {
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(weights.ndimension() == 1, "weights must be 1D");

    int seq_len = x.size(0);
    int d_model = x.size(1);
    torch::Tensor output = torch::empty({seq_len, d_model}, x.options());
    gpt2_transformer_block_cuda(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        weights.data_ptr<float>(), 
        seq_len);
    return output;
}
