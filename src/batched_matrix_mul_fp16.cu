#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>


/*
Implement a batched matrix multiplication in FP16. Given a batch of matrices A of shape [B, M, K] and a batch of matrices B of shape [B, K, N], compute the output batch C of shape [B, M, N] such that for each batch index b:
All matrices are stored in row-major order and use 16-bit floating point numbers (FP16/half). Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
Accumulation during multiplication should use FP32 for better precision before converting the final result to FP16
The final result must be stored in the C array as half
Example 1:
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
Constraints
1 ≤ B ≤ 128
1 ≤ M, N, K ≤ 1024
*/


namespace { 
    constexpr int TILE_SIZE = 32;
}


__global__ void batched_matrix_mul_fp16_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    __shared__ half s_a[TILE_SIZE * TILE_SIZE];
    __shared__ half s_b[TILE_SIZE * TILE_SIZE];

    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int row_c = blockIdx.y * TILE_SIZE + ty;
    const int col_c = blockIdx.x * TILE_SIZE + tx;
    
    float acc = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // load A and B tiles into shared memory
        for (int  i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_a = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col_a = k + i % TILE_SIZE;
            s_a[i] = (row_a < M && col_a < K) ? A[blockIdx.z * M * K + row_a * K + col_a] : __float2half(0.0f);
        }
        for (int  i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row_b = k + i / TILE_SIZE;
            int col_b = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_b[i] = (row_b < K && col_b < N) ? B[blockIdx.z * K * N + row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();

        if (row_c < M && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += __half2float(s_a[ty * TILE_SIZE + i]) * __half2float(s_b[i * TILE_SIZE + tx]);
            }
        }

        // compute partial product
        __syncthreads();
    }
    // write back to C
    if (row_c < M && col_c < N) {
        C[blockIdx.z * M * N + row_c * N + col_c] = __float2half(acc);
    }
}


void batched_matrix_mul_naive_fp16(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(M, TILE_SIZE), BATCH);
    batched_matrix_mul_fp16_kernel<<<grid, block>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


template <int BM, int BN, int BK, int WM, int WN>
__global__ void batched_matrix_mul_warp_wmma_fp16_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) { 
    using namespace nvcuda;

    const int warp_id = threadIdx.x / 32;
    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    __shared__ half s_a[BM * BK];
    __shared__ half s_b[BK * BN];
    __shared__ float s_c[BM * BN];

    half* a_ptr = s_a + (warp_row_id * WM) * BK;
    half* b_ptr = s_b + warp_col_id * WN;
    float* c_ptr = s_c + (warp_row_id * WM) * BN + warp_col_id * WN;

    using FragType_A = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragType_B = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FragType_C = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragType_A frag_a;
    FragType_B frag_b;
    FragType_C frag_acc;

    wmma::fill_fragment(frag_acc, 0.0f);

    for (int k = 0; k < K; k += BK) { 
        // load A and B from global to shared
        for (int i = threadIdx.x; i < BK * BM; i += blockDim.x) {
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[i] = (row_a < M && col_a < K) ? A[blockIdx.z * M * K + row_a * K + col_a] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[i] = (row_b < K && col_b < N) ? B[blockIdx.z * K * N + row_b * N + col_b] : __float2half(0.0f);
        }
        __syncthreads();
        
        // compute wmma
        wmma::load_matrix_sync(frag_a, a_ptr, BK);
        wmma::load_matrix_sync(frag_b, b_ptr, BN);

        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        __syncthreads();
    }

    // write back to C
    wmma::store_matrix_sync(c_ptr, frag_acc, BN, wmma::mem_row_major);
    __syncthreads();
    
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row_c = blockIdx.y * BM + i / BN;
        int col_c = blockIdx.x * BN + i % BN;
        if (row_c < M && col_c < N) {
            C[blockIdx.z * M * N + row_c * N + col_c] = __float2half(s_c[i]);
        }
    }
}

void batched_matrix_mul_warp_wmma_fp16(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int WM = 16;
    constexpr int WN = 16;

    dim3 block(BM * BN / (WN * WM) * 32);
    dim3 grid(div_up(N, BN), div_up(M, BM), BATCH);
    batched_matrix_mul_warp_wmma_fp16_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(A, B, C, BATCH, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


namespace wmma_matmul_fp16 {

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

// copyA 和 copyB 的线程分布
constexpr int COPY_A_THREADS_X = 16;
constexpr int COPY_A_THREADS_Y = BLOCK_SIZE / COPY_A_THREADS_X; // 16
constexpr int COPY_B_THREADS_X = 32;
constexpr int COPY_B_THREADS_Y = BLOCK_SIZE / COPY_B_THREADS_X; // 8

constexpr int THREADS_C_WARP_X = WMMA_N; // 16
constexpr int THREADS_C_WARP_Y = 32 / THREADS_C_WARP_X; // 2


// ====================== 批量矩阵乘法 核心 CUDA Kernel ======================
/**
 * 批量矩阵乘法
 * @param A 输入矩阵，行主序，[batch, M, K]
 * @param B 输入矩阵，行主序，[batch, K, N]
 * @return 输出矩阵，行主序，[batch, M, N]
 */
 __global__ void matmul_fp16_wmma_kernel(const half* A, const half* B, half* C, int M, int N, int K)
{
    using namespace nvcuda;

    // 1. 计算当前 block 负责的 tile 起始位置
    int row0 = blockIdx.y * BLOCK_M;
    int col0 = blockIdx.x * BLOCK_N;
    const half* A_ptr = A + blockIdx.z * M * K; // 当前 batch 的 A 矩阵起始地址
    const half* B_ptr = B + blockIdx.z * K * N; // 当前 batch 的 B 矩阵起始地址
    half* C_ptr = C + blockIdx.z * M * N; // 当前 batch 的 C 矩阵起始地址

    // 2. 共享内存声明，双缓冲
    __shared__ __half sA[2][BLOCK_M][BLOCK_K]; 
    __shared__ __half sB[2][BLOCK_K][BLOCK_N]; 
    __shared__ float sC[BLOCK_M][BLOCK_N];

    // 3. 线程 ID 和 warp ID 计算
    const int warp_id = threadIdx.x >> 5; // 每 32 个线程一个 warp
    const int lane_id = threadIdx.x & 0x1F; // warp 内的线程 ID
    const int warp_x = warp_id & (WARP_X_DIM - 1); // warp 在 tile 中的 x 维位置
    const int warp_y = warp_id / WARP_X_DIM; // warp 在 tile 中的 y 维位置
    // copyA 的线程排布 
    const int tAx = threadIdx.x & (COPY_A_THREADS_X - 1); // 0~15，决定了 copyA 中的 x 维偏移
    const int tAy = threadIdx.x / COPY_A_THREADS_X; // 0~15，决定了 copyA 中的 y 维偏移
    // copyB 的线程排布
    const int tBx = threadIdx.x & (COPY_B_THREADS_X - 1); // 0~31，决定了 copyB 中的 x 维偏移
    const int tBy = threadIdx.x / COPY_B_THREADS_X; // 0~7，决定了 copyB 中的 y 维偏移

    int buffer_id = 0; // 双缓冲切换 ID

    // 4. 寄存器声明
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fragA[BLOCK_M_LOOPS];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fragB[BLOCK_N_LOOPS];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC[BLOCK_M_LOOPS][BLOCK_N_LOOPS];

    // 5. 初始化累加器
    #pragma unroll
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
            sA[0][j + tAy][i + tAx] = (r < M && c < K) ? A_ptr[r * K + c] : CUDART_ZERO_FP16;
        }
    }

    // 预加载 B矩阵
    #pragma unroll
    for (int i = 0; i < BLOCK_N; i += COPY_B_THREADS_X) { 
        int c = col0 + i + tBx; // N 维的全局索引
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += COPY_B_THREADS_Y) {
            int r = 0 + j + tBy; // K 维的全局索引
            sB[0][j + tBy][(i + tBx)] = (r < K && c < N) ? B_ptr[r * N + c] : CUDART_ZERO_FP16;
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
                    sA[buffer_id ^ 1][j + tAy][i + tAx] = (r < M && c < K) ? A_ptr[r * K + c] : CUDART_ZERO_FP16;
                }
            }

            // 预加载 B矩阵
            #pragma unroll
            for (int i = 0; i < BLOCK_N; i += COPY_B_THREADS_X) { 
                int c = col0 + i + tBx; // N 维的全局索引
                #pragma unroll
                for (int j = 0; j < BLOCK_K; j += COPY_B_THREADS_Y) {
                    int r = k + j + tBy; // K 维的全局索引
                    sB[buffer_id ^ 1][j + tBy][(i + tBx)] = (r < K && c < N) ? B_ptr[r * N + c] : CUDART_ZERO_FP16;
                }
            }
            __syncthreads();
        }
        // 从共享内存加载数据到寄存器
        // 加载 A矩阵
        #pragma unroll
        for (int i = 0; i < BLOCK_M_LOOPS; i++) {
            int r = i * BLOCK_M_TILES + warp_y * WMMA_M;
            wmma::load_matrix_sync(fragA[i], &sA[buffer_id][r][0], BLOCK_K);
        }
        // 加载 B矩阵
        #pragma unroll
        for (int i = 0; i < BLOCK_N_LOOPS; i++) {
            int c = i * BLOCK_N_TILES + warp_x * WMMA_N;
            wmma::load_matrix_sync(fragB[i], &sB[buffer_id][0][c], BLOCK_N);
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
            wmma::store_matrix_sync(&sC[r][c], fragC[i][j], BLOCK_N, wmma::mem_row_major);
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
                        C_ptr[(row0 + row) * N + (col0 + col)] = __float2half(sC[row][col]);
                    } 
                }
            }
        }
    }
}



// ====================== 对外调用函数 (签名完全不变) ======================
void launch_matmul_fp16_wmma_kernel(const half* A, const half* B, half* C, int BATCH, int M, int N, int K)
{
    printf("launch_matmul_fp16_wmma_kernel\n");
    dim3 block(BLOCK_SIZE);
    dim3 grid(div_up(N, BLOCK_N), div_up(M, BLOCK_M), BATCH);
    matmul_fp16_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace wmma_matmul_fp16



// A, B, C are device pointers
void batched_matrix_mul_fp16_cuda(const half* A, const half* B, half* C, int BATCH, int M, int N, int K) {
    // batched_matrix_mul_naive_fp16(A, B, C, BATCH, M, N, K);
    // batched_matrix_mul_warp_wmma_fp16(A, B, C, BATCH, M, N, K);
    wmma_matmul_fp16::launch_matmul_fp16_wmma_kernel(A, B, C, BATCH, M, N, K);
}


torch::Tensor batched_matrix_mul_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 3, "Input A must be 3D");
    TORCH_CHECK(B.ndimension() == 3, "Input B must be 3D    ");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have compatible shapes");
    TORCH_CHECK(A.size(2) == B.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be of type float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be of type float16");

    int BATCH = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    torch::Tensor C = torch::empty({BATCH, M, N}, A.options());
    batched_matrix_mul_fp16_cuda(
        reinterpret_cast<half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        BATCH, M, N, K
    );
    return C;
}