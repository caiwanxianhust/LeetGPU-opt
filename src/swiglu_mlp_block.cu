#include "utils.cuh"


namespace gemm_pipline {

__host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// 定义宏，用于将 float 数组强制转换为 float4 类型，实现向量化访问
#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

// --- 核心配置参数 ---
constexpr int BLOCK_M = 128;      // Block 在 M 维度（行）处理的大小
constexpr int BLOCK_N = 128;      // Block 在 N 维度（列）处理的大小
constexpr int BLOCK_K = 8;        // Block 在 K 维度（深度）处理的大小
constexpr int BLOCK_SIZE = 256;   // 每个 Block 的线程总数

// --- 线程布局配置 
// A 矩阵加载时的线程布局：8列 x 32行
constexpr int THREAD_A_LAYOUT_X = 8;
constexpr int THREAD_A_LAYOUT_Y = BLOCK_SIZE / THREAD_A_LAYOUT_X; // 32

// B 矩阵加载时的线程布局：32列 x 8行
constexpr int THREAD_B_LAYOUT_X = 32;
constexpr int THREAD_B_LAYOUT_Y = BLOCK_SIZE / THREAD_B_LAYOUT_X; // 8

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
 

/**
 * GEMM 矩阵乘法 
 * @param A 输入矩阵 A, 矩阵大小为 M x K
 * @param B 输入矩阵 B, 矩阵大小为 K x N
 * @param C 输出矩阵 C, 矩阵大小为 M x N
 */
__global__ void gemm_fp32_pipeline_kernel(const float* A, const float* B, float* C,
    int M, int N, int K) 
{   
    // 1. 计算当前 Block 的起始索引
    const int row0 = blockIdx.y * BLOCK_M;
    const int col0 = blockIdx.x * BLOCK_N;

    // 2. 共享内存声明，使用双缓冲
    __shared__ float sA[2][BLOCK_K][BLOCK_M];   // 转置存储以优化访问
    __shared__ float sB[2][BLOCK_K][BLOCK_N];

    // 3. 线程 ID 映射
    // copy A 按 8列 x 32行的方式映射线程
    const int tAx = threadIdx.x & (THREAD_A_LAYOUT_X - 1);  // 0 ~ 7
    const int tAy = threadIdx.x / THREAD_A_LAYOUT_X;    // 0 ~ 31

    // copy B 按 32列 x 8行的方式映射线程
    const int tBx = threadIdx.x & (THREAD_B_LAYOUT_X - 1);  // 0 ~ 31
    const int tBy = threadIdx.x / THREAD_B_LAYOUT_X;    //  0 ~ 7

    // C 矩阵计算的线程映射：每个线程负责计算 C 中一个 8x8 的子块
    const int warpId = (threadIdx.x >> 5);  // 0 ~ 7
    const int laneId = (threadIdx.x & 31);  // 0 ~ 31

    // 将 Warp 重新组织为 2D 网格
    int warpx = warpId & (THREAD_C_WARP_DIM_X - 1); // Warp 在 X 方向的 ID, 0 ~ 1
    int warpy = warpId / THREAD_C_WARP_DIM_X;       // Warp 在 Y 方向的 ID, 0 ~ 3

    // 将 Lane 重新组织为 2D 网格 (这里使用了特殊的位运算映射，为了配合 XOR Bank Conflict 消除)
    const int lanex = (laneId & 15) >> 1; // 取低 4 位右移 1 位 -> 0-7
    const int laney = ((laneId >> 4) << 1) + (laneId & 1); // 取最高位左移 1 + 最低位 -> 0-3

    // 计算当前线程在 C 矩阵计算网格中的最终坐标
    const int tCx = warpx * THREAD_C_WARP_X + lanex;    // 0 ~ 15
    const int tCy = warpy * THREAD_C_WARP_Y + laney;    // 0 ~ 15

    // 4. 寄存器声明
    // acc: 每个线程维护一个 8x8 的累加器矩阵，用于存储计算结果
    float acc[THREAD_C_Y_TILE_SIZE][THREAD_C_X_TILE_SIZE] = {0.0f};
    // tCsA, tCsB: 寄存器缓存，用于存储从 Shared Memory 加载的数据
    // [2][...] 用于双缓冲流水线
    float tCsA[2][THREAD_C_Y_TILE_SIZE];
    float tCsB[2][THREAD_C_X_TILE_SIZE];

    int bufferId = 0; // 双缓冲索引 (0 或 1)

    // ==========================================
    // 5. 预加载阶段 (加载 K=0 的数据块)
    // ==========================================
    // 加载 A 矩阵的第一块 (BLOCK_K x BLOCK_M)
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i+=THREAD_A_LAYOUT_Y) {
        int r = row0 + i + tAy;
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j+=THREAD_A_LAYOUT_X) {    
            int c = 0 + j + tAx;
            sA[0][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
    }
    // 加载 B 矩阵的第一块 (BLOCK_K x BLOCK_N)
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i+=THREAD_B_LAYOUT_Y) {
        int r = 0 + i + tBy;
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j+=THREAD_B_LAYOUT_X) {
            int c = col0 + j + tBx;
            sB[0][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
    }
    __syncthreads();

    // 6. 主循环：细粒度流水线
    for (int k = BLOCK_K; k < K + BLOCK_K; k += BLOCK_K) {
        // 6.1 计算当前块的乘积
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
            
            // 6.2 预加载下一块数据到寄存器
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
                    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }

        // ==========================================
        // 7. 预加载下一块 K 数据 (Global -> Shared)
        // ==========================================
        if (k < K) {
            // 加载 A 到 Shared Memory
            // 加载 A 矩阵 (BLOCK_K x BLOCK_M)
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i+=THREAD_A_LAYOUT_Y) {
                int r = row0 + i + tAy;
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j+=THREAD_A_LAYOUT_X) {    
                    int c = k + j + tAx;
                    sA[bufferId ^ 1][j + tAx][(i + tAy) ^ ((j + tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
                }
            }
            // 加载 B 矩阵 (BLOCK_K x BLOCK_N)
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i+=THREAD_B_LAYOUT_Y) {
                int r = k + i + tBy;
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j+=THREAD_B_LAYOUT_X) {
                    int c = col0 + j + tBx;
                    sB[bufferId ^ 1][i + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
                }
            }
            __syncthreads();
        }
        bufferId ^= 1; // 切换双缓冲索引 (0 -> 1, 1 -> 0)
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
                C[r * N + c] = acc[i][j];
            }
        }
    }
}


void launch_gemm_pipline_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    printf("launch_gemm_pipline_kernel\n");
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M));
    gemm_fp32_pipeline_kernel<<<grid, block>>>(A, B, C, M, N, K);
}


__device__ __forceinline__  float silu(float val) {
    return val / (1 + __expf(-1.0f * val));
}

__global__ void silu_and_elem_mul_kernel(const float* xg, const float* xu, float* output, int M, int d_ffn) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < M * d_ffn; i += blockDim.x * gridDim.x) {
        output[i] = silu(xg[i]) * xu[i];
    }
}


void launch_silu_and_elem_mul_kernel(const float* xg, const float* xu, float* output, int M, int d_ffn) {
    constexpr int block_size = 256;
    int grid_size = 1024;
    if (M * d_ffn <= block_size * 1024) {
        grid_size = div_up(M * d_ffn, block_size);
    }
    silu_and_elem_mul_kernel<<<grid_size, block_size>>>(xg, xu, output, M, d_ffn);
}



// x, W_gate, W_up, W_down, output are device pointers
void swiglu_mlp_block_gemm_pipeline_cuda(
    const float* x, const float* W_gate, const float* W_up, const float* W_down,
    float* output, int M, int d_model, int d_ffn) {
    
    float* d_buf;
    size_t buf_size = M * d_ffn * sizeof(float) * 3;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    float* d_xg = d_buf;
    float* d_xu = d_xg + M * d_ffn;
    float* d_gu = d_xu + M * d_ffn;

    launch_gemm_pipline_kernel(x, W_gate, d_xg, M, d_ffn, d_model);
    launch_gemm_pipline_kernel(x, W_up, d_xu, M, d_ffn, d_model);
    launch_silu_and_elem_mul_kernel(d_xg, d_xu, d_gu, M, d_ffn);
    launch_gemm_pipline_kernel(d_gu, W_down, output, M, d_model, d_ffn);

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


}   // namespace gemm_pipline



// x, W_gate, W_up, W_down, output are device pointers
void swiglu_mlp_block_cuda(const float* x, const float* W_gate, const float* W_up, const float* W_down,
    float* output, int M, int d_model, int d_ffn) {
    gemm_pipline::swiglu_mlp_block_gemm_pipeline_cuda(x, W_gate, W_up, W_down, output, M, d_model, d_ffn);
}