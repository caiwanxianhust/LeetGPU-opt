#include "utils.cuh"

#include <cuda_fp16.h>
#include <stdint.h>
#include <mma.h>
#include <torch/extension.h>


using namespace nvcuda;

/*
Implement a weight-only INT4 quantized matrix multiplication (W4A16), a core kernel used in modern LLM inference. Given a float16 activation matrix x of shape M × K and a weight matrix stored in packed INT4 format, compute the output matrix y = x × WT of shape M × N, where W is the dequantized float16 weight matrix of shape N × K.

Packing format: Each byte of w_q stores two INT4 weights. The high nibble (bits 7–4) holds weight w[n, 2i] and the low nibble (bits 3–0) holds w[n, 2i+1]. INT4 values are stored unsigned in the range [0, 15] with an offset of 8, so the signed weight is nibble − 8, giving values in [−8, 7].

Dequantization: Weights are dequantized group-wise. Each contiguous block of group_size weights along the K dimension shares one float16 scale:

W[n, k] = (w_q_nibble[n, k] - 8) * scales[n, k // group_size]
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in y

Constraints
1 ≤ M, N ≤ 8,192
1 ≤ K ≤ 8,192
K is divisible by 2 and by group_size
group_size ∈ {2, 4, 8, 16, 32, 64, 128}
All tensors are stored in row-major order
Input dtype: x and scales are float16; w_q is uint8
Output dtype: y is float16
Performance is measured with M = 4,096, N = 4,096, K = 4,096, group_size = 128
*/



static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])


namespace wmma_w4a16 {

// WMMA 形状
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// CTA 分块
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = WMMA_K;

constexpr int BLOCK_SIZE = 256; // 每个 block 启动的线程数

// warp 按 4行x2列分布
constexpr int WARP_X_DIM = 2;
constexpr int WARP_Y_DIM = (BLOCK_SIZE >> 5) / WARP_X_DIM;

// 一个 block 一次性处理的 tile 大小
constexpr int BLOCK_M_TILES = WARP_Y_DIM * WMMA_M; // 16 行 x (256/32/2) = 64 行
constexpr int BLOCK_N_TILES = WARP_X_DIM * WMMA_N; // 16 列 x 2 = 32 列

// block 在 M 和 N 维度上的循环次数
constexpr int BLOCK_M_LOOPS = BLOCK_M / BLOCK_M_TILES; // 2
constexpr int BLOCK_N_LOOPS = BLOCK_N / BLOCK_N_TILES; // 2

constexpr int SMEM_A_PAD = 0;
constexpr int SMEM_B_PAD = 0;
constexpr int SMEM_C_PAD = 0;

// copyA 和 copyB 的线程分布
constexpr int COPY_A_THREADS_X = 16;
constexpr int COPY_A_THREADS_Y = BLOCK_SIZE / COPY_A_THREADS_X; // 16
constexpr int COPY_B_THREADS_X = 8;
constexpr int COPY_B_THREADS_Y = BLOCK_SIZE / COPY_B_THREADS_X; // 32

constexpr int THREADS_C_WARP_X = WMMA_N; // 16
constexpr int THREADS_C_WARP_Y = 32 / THREADS_C_WARP_X; // 2


// ====================== INT4 反量化函数 (你提供的原版) ======================
static __device__ half2 dequantize_int4(uint8_t packed_byte, __half scale) {
    // 1. 解包两个 INT4 (0~15)
    int high_nibble = (packed_byte >> 4) & 0x0F; // w[n, 2i]
    int low_nibble  = packed_byte & 0x0F;        // w[n, 2i+1]

    // 2. 转换为有符号 INT4 (-8~7)
    float w0_s = static_cast<float>(high_nibble - 8) * __half2float(scale);
    float w1_s = static_cast<float>(low_nibble  - 8) * __half2float(scale);

    __half h0 = __float2half(w0_s);
    __half h1 = __float2half(w1_s);

    // 正确构造 __half2
    __half2 w2 = __halves2half2(h0, h1);
    
    return w2;
}


// ====================== W4A16 核心 CUDA Kernel ======================
/**
 * W4A16 矩阵乘法
 * @param x 输入矩阵，行主序，[M, K]
 * @param w_q 输入矩阵，行主序，[N, K / 2]
 * @param scales 输入矩阵，行主序，[N, K / group_size]
 * @return 输出矩阵，行主序，[M, N]
 */
 __global__ void int4_weight_only_quantized_matMul_wmma_kernel(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y,
    int M, int N, int K, int group_size)
{
    using namespace nvcuda;

    // 1. 计算当前 block 负责的 tile 起始位置
    int row0 = blockIdx.y * BLOCK_M;
    int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明，双缓冲
    __shared__ __half sA[2][BLOCK_M][BLOCK_K + SMEM_A_PAD]; // x 的 tile，带 padding 防止 bank conflict
    __shared__ __half sB[2][BLOCK_N][BLOCK_K + SMEM_B_PAD]; // w 的 tile，带 padding 防止 bank conflict
    __shared__ float sC[BLOCK_M][BLOCK_N + SMEM_C_PAD];

    // 3. 线程 ID 和 warp ID 计算
    const int warp_id = threadIdx.x >> 5; // 每 32 个线程一个 warp
    const int lane_id = threadIdx.x & 0x1F; // warp 内的线程 ID
    const int warp_x = warp_id % WARP_X_DIM; // warp 在 tile 中的 x 维位置
    const int warp_y = warp_id / WARP_X_DIM; // warp 在 tile 中的 y 维位置
    // copyA 的线程排布 
    const int tAx = threadIdx.x & (COPY_A_THREADS_X - 1); // 0~15，决定了 copyA 中的 x 维偏移
    const int tAy = threadIdx.x / COPY_A_THREADS_X; // 0~15，决定了 copyA 中的 y 维偏移
    // copyB 的线程排布
    const int tBx = threadIdx.x & (COPY_B_THREADS_X - 1); // 0~7，决定了 copyB 中的 x 维偏移
    const int tBy = threadIdx.x / COPY_B_THREADS_X; // 0~31，决定了 copyB 中的 y 维偏移

    int buffer_id = 0; // 双缓冲切换 ID

    // 4. 寄存器声明
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fragA[BLOCK_M_LOOPS];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> fragB[BLOCK_N_LOOPS];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC[BLOCK_M_LOOPS][BLOCK_N_LOOPS];

    // 5. 初始化累加器
    for (int i = 0; i < BLOCK_M_LOOPS; i++) {
        for (int j = 0; j < BLOCK_N_LOOPS; j++) {
            wmma::fill_fragment(fragC[i][j], 0.0f);
        }   
    }

    // 6. 预加载第一块数据到共享内存
    // 预加载 A矩阵
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += COPY_A_THREADS_X) {
        int c = 0 + i + tAx; // K 维的全局索引
        #pragma unroll
        for (int j = 0; j < BLOCK_M; j += COPY_A_THREADS_Y) {
            int r = row0 + j + tAy; // M 维的全局索引
            sA[0][j + tAy][i + tAx] = (r < M && c < K) ? x[r * K + c] : CUDART_ZERO_FP16;
        }
    }

    // 预加载 B矩阵
    #pragma unroll
    for (int i = 0; i < (BLOCK_K / 2); i += COPY_B_THREADS_X) { // 每个线程处理 2 个连续的 K 元素，因为每个 uint8 包含两个 INT4 权重
        int c = 0 + i + tBx; // K 维的全局索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += COPY_B_THREADS_Y) {
            int r = col0 + j + tBy; // N 维的全局索引
            // 计算 scale 索引
            int scale_idx = r * (K / group_size) + (c * 2) / group_size; // 每 group_size 个权重共享一个 scale，注意每个 uint8 包含两个权重
            __half scale = scales[scale_idx];
            if (r < N && c < (K / 2)) {
                uint8_t packed_byte = w_q[r * (K / 2) + c]; // 每个 uint8 包含两个 INT4 权重
                half2 w2 = dequantize_int4(packed_byte, scale);
                __half w0 = __low2half(w2);   //  低 16 位
                __half w1 = __high2half(w2);  //  高 16 位
                sB[0][j + tBy][(i + tBx) << 1] = w0; // w[n, 2i]
                sB[0][j + tBy][((i + tBx) << 1) + 1] = w1; // w[n, 2i+1]
            }
            else {
                sB[0][j + tBy][(i + tBx) << 1] = CUDART_ZERO_FP16;
                sB[0][j + tBy][((i + tBx) << 1) + 1] = CUDART_ZERO_FP16;
            }
        }
    }
    __syncthreads();

    // 7. 主循环，双缓冲加载和计算交错进行
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 预加载下一块数据到共享内存
        if (k < K) { 
            // 加载 A矩阵
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += COPY_A_THREADS_X) {
                int c = k + i + tAx; // K 维的全局索引
                #pragma unroll
                for (int j = 0; j < BLOCK_M; j += COPY_A_THREADS_Y) {
                    int r = row0 + j + tAy; // M 维的全局索引
                    sA[buffer_id ^ 1][j + tAy][i + tAx] = (r < M && c < K) ? x[r * K + c] : CUDART_ZERO_FP16;
                }
            }

            // 预加载 B矩阵
            #pragma unroll
            for (int i = 0; i < (BLOCK_K / 2); i += COPY_B_THREADS_X) { // 每个线程处理 2 个连续的 K 元素，因为每个 uint8 包含两个 INT4 权重
                int c = (k / 2) + i + tBx; // K 维的全局索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += COPY_B_THREADS_Y) {
                    int r = col0 + j + tBy; // N 维的全局索引
                    // 计算 scale 索引
                    int scale_idx = r * (K / group_size) + (c * 2) / group_size; // 每 group_size 个权重共享一个 scale，注意每个 uint8 包含两个权重
                    __half scale = scales[scale_idx];
                    if (r < N && c < (K / 2)) {
                        uint8_t packed_byte = w_q[r * (K / 2) + c]; // 每个 uint8 包含两个 INT4 权重
                        
                        half2 w2 = dequantize_int4(packed_byte, scale);
                        __half w0 = __low2half(w2);   // 低 16 位
                        __half w1 = __high2half(w2);  // 高 16 位
                        sB[buffer_id ^ 1][j + tBy][(i + tBx) << 1] = w0; // w[n, 2i]
                        sB[buffer_id ^ 1][j + tBy][((i + tBx) << 1) + 1] = w1; // w[n, 2i+1]
                    }
                    else {
                        sB[buffer_id ^ 1][j + tBy][(i + tBx) << 1] = CUDART_ZERO_FP16;
                        sB[buffer_id ^ 1][j + tBy][((i + tBx) << 1) + 1] = CUDART_ZERO_FP16;
                    }
                }
            }
        }
        // 从共享内存加载数据到寄存器
        // 加载 A矩阵
        #pragma unroll
        for (int i = 0; i < BLOCK_M_LOOPS; i++) {
            int r = i * BLOCK_M_TILES + warp_y * WMMA_M;
            wmma::load_matrix_sync(fragA[i], &sA[buffer_id][r][0], BLOCK_K + SMEM_A_PAD);
        }
        // 加载 B矩阵
        #pragma unroll
        for (int i = 0; i < BLOCK_N_LOOPS; i++) {
            int c = i * BLOCK_N_TILES + warp_x * WMMA_N;
            wmma::load_matrix_sync(fragB[i], &sB[buffer_id][c][0], BLOCK_K + SMEM_B_PAD);
        }

        // Matrix multiply-accumulate
        #pragma unroll
        for (int i = 0; i < BLOCK_M_LOOPS; i++) { 
            #pragma unroll
            for (int j = 0; j < BLOCK_N_LOOPS; j++) {
                wmma::mma_sync(fragC[i][j], fragA[i], fragB[j], fragC[i][j]);
            }
        }
        __syncthreads();

        buffer_id ^= 1;
    }

    // 8. 写回计算结果到sC
    #pragma unroll
    for (int i = 0; i < BLOCK_M_LOOPS; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N_LOOPS; j++) {
            int r = i * BLOCK_M_TILES + warp_y * WMMA_M;
            int c = j * BLOCK_N_TILES + warp_x * WMMA_N;
            wmma::store_matrix_sync(&sC[r][c], fragC[i][j], BLOCK_N + SMEM_C_PAD, wmma::mem_row_major);
        }
    }
    // 9. 读出计算结果到全局内存
    int lanex = lane_id & (THREADS_C_WARP_X - 1); // 0-15
    int laney = lane_id / THREADS_C_WARP_X; // 0-1
    #pragma unroll
    for (int i = 0; i < BLOCK_M_LOOPS; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N_LOOPS; j++) {
            #pragma unroll
            for (int tn = 0; tn < WMMA_N; tn += THREADS_C_WARP_X) {
                int col = j * BLOCK_N_TILES + warp_x * WMMA_N + tn + lanex;
                #pragma unroll
                for (int tm = 0; tm < WMMA_M; tm += THREADS_C_WARP_Y) {
                    int row = i * BLOCK_M_TILES + warp_y * WMMA_M + tm + laney;
                    if ((row0 + row) < M && (col0 + col) < N) {
                        y[(row0 + row) * N + (col0 + col)] = __float2half(sC[row][col]);
                    } 
                }
            }
        }
    }
}



// ====================== 对外调用函数 (签名完全不变) ======================
void int4_weight_only_quantized_matMul_wmma_cuda(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
    int N, int K, int group_size)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    int4_weight_only_quantized_matMul_wmma_kernel<<<grid, block>>>(x, w_q, scales, y, M, N, K, group_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace wmma_w4a16


namespace naive_w4a16 {

// --- 核心配置参数 ---
constexpr int BLOCK_M = 128;      // Block 在 M 维度（行）处理的大小
constexpr int BLOCK_N = 128;      // Block 在 N 维度（列）处理的大小
constexpr int BLOCK_K = 8;        // Block 在 K 维度（深度）处理的大小
constexpr int BLOCK_SIZE = 256;   // 每个 Block 的线程总数

// --- 线程布局配置 (这是该 Kernel 最精妙的部分) ---
// A 矩阵加载时的线程布局：8列 x 32行
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X; // 32

// B 矩阵加载时的线程布局：4列 x 64行
constexpr int THREAD_B_T_LAYOUT_X = 4;
constexpr int THREAD_B_T_LAYOUT_Y = BLOCK_SIZE / THREAD_B_T_LAYOUT_X; // 64

// C 矩阵（结果）计算时的线程布局：16列 x 16行
constexpr int THREAD_C_LAYOUT_X = 16;
constexpr int THREAD_C_LAYOUT_Y = BLOCK_SIZE / THREAD_C_LAYOUT_X; // 16

// 每个线程负责计算的 C 矩阵分块大小
constexpr int THREAD_C_X_TILE_SIZE = BLOCK_N / THREAD_C_LAYOUT_X; // 128 / 16 = 8
constexpr int THREAD_C_Y_TILE_SIZE = BLOCK_M / THREAD_C_LAYOUT_Y; // 128 / 16 = 8

// Warp 内部细分：在 block 16列 x 16行内将 32 线程的 Warp 分为 4x8 的结构，warp 按照 4 x 2 排布
constexpr int THREAD_C_WARP_X = 8;
constexpr int THREAD_C_WARP_Y = 32 / THREAD_C_WARP_X; // 4
constexpr int THREAD_C_WARP_DIM_X = THREAD_C_LAYOUT_X / THREAD_C_WARP_X; // 2


// ====================== INT4 反量化函数 (你提供的原版) ======================
static __device__ float2 dequantize_int4_to_fp32(uint8_t packed_byte, __half scale) {
    // 1. 解包两个 INT4 (0~15)
    int high_nibble = (packed_byte >> 4) & 0x0F; // w[n, 2i]
    int low_nibble  = packed_byte & 0x0F;        // w[n, 2i+1]

    // 2. 转换为有符号 INT4 (-8~7)
    float w0_s = static_cast<float>(high_nibble - 8) * __half2float(scale);
    float w1_s = static_cast<float>(low_nibble  - 8) * __half2float(scale);
    
    return make_float2(w0_s, w1_s);
}




static __global__ void int4_weight_only_quantized_matMul_naive_kernel(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
    int N, int K, int group_size
) {
    // 1、当前 block 负责计算 C 矩阵中以 (row0, col0) 为左上角，大小为 BLOCK_M x BLOCK_N 的子矩阵
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明
    __shared__ float sA[2][BLOCK_K][BLOCK_M]; // 双缓冲 A 矩阵，大小为 16 x 128，转置存储
    __shared__ float sB[2][BLOCK_K][BLOCK_N]; // 双缓冲 B 矩阵，大小为 16 x 128，转置存储

    // 3. 线程ID和线程布局计算
    // copyA 线程布局：8列 x 32行
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1); // 0-7 列索引
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X; // 0-31 行索引
    // copyB 线程布局：4列 x 64行
    const int tBx = threadIdx.x & (THREAD_B_T_LAYOUT_X - 1); // 0-3 列索引
    const int tBy = threadIdx.x / THREAD_B_T_LAYOUT_X; // 0-64 行索引
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
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? __half2float(x[r * K + c]) : 0.0f;
        }
    }

    // 加载 B 的第一块
    #pragma unroll
    for (int i = 0; i < (BLOCK_K / 2); i += THREAD_B_T_LAYOUT_X) { // 每个线程处理 2 个连续的 K 元素，因为每个 uint8 包含两个 INT4 权重
        int c = 0 + i + tBx; // K 维的全局索引
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += THREAD_B_T_LAYOUT_Y) {
            int r = col0 + j + tBy; // N 维的全局索引
            // 计算 scale 索引
            int scale_idx = r * (K / group_size) + (c * 2) / group_size; // 每 group_size 个权重共享一个 scale，注意每个 uint8 包含两个权重
            __half scale = scales[scale_idx];
            if (r < N && c < (K / 2)) {
                uint8_t packed_byte = w_q[r * (K / 2) + c]; // 每个 uint8 包含两个 INT4 权重
                float2 w2 = dequantize_int4_to_fp32(packed_byte, scale);
                sB[0][(i + tBx) << 1][(j + tBy) ^ (((i + tBx) << 1) << 2)] = w2.x; // w[n, 2i]
                sB[0][((i + tBx) << 1) + 1][(j + tBy) ^ ((((i + tBx) << 1) + 1) << 2)] = w2.y; // w[n, 2i+1]
            }
            else {
                sB[0][(i + tBx) << 1][(j + tBy) ^ (((i + tBx) << 1) << 2)] = 0.0f;
                sB[0][((i + tBx) << 1) + 1][(j + tBy) ^ ((((i + tBx) << 1) + 1) << 2)] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 6. 主循环
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的矩阵乘法
        #pragma unroll
        for (int tk = 0; tk < BLOCK_K + 1; tk++) {
            // 计算 mma
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
                #pragma unroll
                // 加载 A 的分块到寄存器，一次加载 8 个float，并且按照寄存器访问模式进行重排，保证连续访问
                #pragma unroll
                for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 3; tm++) {
                    int r = (tm * THREAD_C_LAYOUT_Y + tCy) << 3;
                    FLOAT4(tCsA[tk & 1][tm << 3]) = FLOAT4(sA[bufferId][tk][r ^ ((tk << 2))]);
                    FLOAT4(tCsA[tk & 1][(tm << 3) + 4]) = FLOAT4(sA[bufferId][tk][(r + 4) ^ ((tk << 2))]);
                }

                // 加载 B 的分块到寄存器，一次加载 8 个 float，并且按照寄存器访问模式进行重排，保证连续访问
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
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += THREAD_A_LAYOUT_X) {
                    int c = k + j + tAx;
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? __half2float(x[r * K + c]) : 0.0f;
                }
            }

            // 加载 B 的第一块
            #pragma unroll
            for (int i = 0; i < (BLOCK_K / 2); i += THREAD_B_T_LAYOUT_X) { // 每个线程处理 2 个连续的 K 元素，因为每个 uint8 包含两个 INT4 权重
                int c = k / 2 + i + tBx; // K 维的全局索引
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j += THREAD_B_T_LAYOUT_Y) {
                    int r = col0 + j + tBy; // N 维的全局索引
                    // 计算 scale 索引
                    int scale_idx = r * (K / group_size) + (c * 2) / group_size; // 每 group_size 个权重共享一个 scale，注意每个 uint8 包含两个权重
                    __half scale = scales[scale_idx];
                    if (r < N && c < (K / 2)) {
                        uint8_t packed_byte = w_q[r * (K / 2) + c]; // 每个 uint8 包含两个 INT4 权重
                        float2 w2 = dequantize_int4_to_fp32(packed_byte, scale);
                        sB[bufferId ^ 1][(i + tBx) << 1][(j + tBy) ^ (((i + tBx) << 1) << 2)] = w2.x; // w[n, 2i]
                        sB[bufferId ^ 1][((i + tBx) << 1) + 1][(j + tBy) ^ ((((i + tBx) << 1) + 1) << 2)] = w2.y; // w[n, 2i+1]
                    }
                    else {
                        sB[bufferId ^ 1][(i + tBx) << 1][(j + tBy) ^ (((i + tBx) << 1) << 2)] = 0.0f;
                        sB[bufferId ^ 1][((i + tBx) << 1) + 1][(j + tBy) ^ ((((i + tBx) << 1) + 1) << 2)] = 0.0f;
                    }
                }
            }
        }
        __syncthreads();
        // 切换缓冲区
        bufferId ^= 1;
    }
    // 7. 将计算结果写回全局内存
    #pragma unroll
    for (int i = 0; i < THREAD_C_Y_TILE_SIZE; i++) {
        int r =  row0 + (tCy << 3) + (i & (~7)) * THREAD_C_LAYOUT_Y + (i & 7);
        #pragma unroll
        for (int j = 0; j < THREAD_C_X_TILE_SIZE; j++) {
            int c = col0 + (tCx << 3) + (j & (~7)) * THREAD_C_LAYOUT_X + (j & 7);
            if (r < M && c < N) {
                y[r * N + c] = __float2half(acc[i][j]);
            }
        }
    }
}


static void int4_weight_only_quantized_matMul_naive_cuda(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
    int N, int K, int group_size)
{
    printf("int4_weight_only_quantized_matMul_naive_cuda\n");
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    int4_weight_only_quantized_matMul_naive_kernel<<<grid, block>>>(x, w_q, scales, y, M, N, K, group_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


} // namespace naive_w4a16




// ====================== 对外调用函数 (签名完全不变) ======================
void int4_weight_only_quantized_matMul_cuda(
    const __half* x, const uint8_t* w_q, const __half* scales, __half* y, int M,
    int N, int K, int group_size)
{
    // wmma_w4a16::int4_weight_only_quantized_matMul_wmma_cuda(x, w_q, scales, y, M, N, K, group_size);
    naive_w4a16::int4_weight_only_quantized_matMul_naive_cuda(x, w_q, scales, y, M, N, K, group_size);
}


torch::Tensor int4_weight_only_quantized_matMul(
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& scales,
    const int group_size
) {
    int M = x.size(0);
    int K = x.size(1);
    int N = w_q.size(0);
    torch::Tensor output = torch::empty({M, N}, x.options());

    int4_weight_only_quantized_matMul_cuda(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_q.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        M, N, K, group_size
    );
    return output;
}