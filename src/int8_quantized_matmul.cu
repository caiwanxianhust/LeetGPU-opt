#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <cstdint>
#include <mma.h>
#include <torch/extension.h>




/*
Implement a quantized matrix multiplication program for 8-bit signed integer matrices. Given two input matrices A of dimensions 
 and B of dimensions 
, quantization scales scale_A, scale_B, output scale scale_C, zero-points zero_point_A, zero_point_B, zero_point_C, compute:
 
where s_A = scale_A, z_A = zero_point_A, etc.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the output matrix C as int8
After accumulation in int32 and scaling in float32, values must be rounded to the nearest integer, shifted by zero_point_C, and clamped to the [-128, 127] range
Example 1:
     Input:
     A = [[1, 2],
          [3, 4]]
     B = [[5, 6],
          [7, 8]]
     M = 2, N = 2, K = 2
     scale_A = 0.1, scale_B = 0.2, scale_C = 0.05
     zero_point_A = 0, zero_point_B = 0, zero_point_C = 0
     
     Output:
     C = [[19, 22],
          [43, 50]]
     
Example 2:
     Input:
     A = [[1, 2]]
     B = [[3],
          [4]]
     M = 1, N = 1, K = 2
     scale_A = 1.0, scale_B = 1.0, scale_C = 1.0
     zero_point_A = 1, zero_point_B = 3, zero_point_C = 5
     
     Output:
     C = [[6]]
     
Constraints
1 ≤ M, N, K ≤ 4096
scale_A, scale_B, scale_C are positive floats
-128 ≤ zero_point_A, zero_point_B, zero_point_C ≤ 127
*/



template<int BM, int BN, int BK>
__global__ void int8_quantized_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    int tc = threadIdx.x % BN;
    int tr = threadIdx.x / BN;
    int col = blockIdx.x * BN + tc;
    int row = blockIdx.y * BM + tr;

    // load tile to smem:
    __shared__ int s_a[BM][BK];
    __shared__ int s_b[BK][BN];

    int numTiles = (K + BK - 1) / BK;
    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + t * BK;
            s_a[i / BK][i % BK] = (col_a < K && row_a < M) ? static_cast<int>(A[row_a * K + col_a]) - zero_point_A : 0;
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row_b = i / BN + t * BK;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[i / BN][i % BN] = (col_b < N && row_b < K) ? static_cast<int>(B[row_b * N + col_b]) - zero_point_B : 0;
        }
        __syncthreads();

        // Find dot product 
        if (col < N && row < M) {
            for (int i = 0; i < BK; i++) {
                acc += s_a[tr][i] * s_b[i][tc];
            }
        }
        __syncthreads();
    }
    if (col < N && row < M) {

        float scaled = (__int2float_rn(acc) * scale_A * scale_B / scale_C);
        acc = min(max(__float2int_rn(scaled) + zero_point_C, -128), 127);

        C[row*N + col] = acc;
    }
}


static __device__ __host__ __forceinline__ int8_t clamp_int8(int x) {
    return static_cast<int8_t>(min(max(x, -128), 127));
}


template <int BM, int BN, int BK, int WM, int WN>
__global__ void int8_quantized_matmul_warp_wmma_kernel(
    const int8_t* A, const int8_t* B, int8_t* C, 
    int M, int N, int K, float scale_A, float scale_B, float scale_C, 
    int zero_point_A, int zero_point_B, int zero_point_C) 
{ 
    using namespace nvcuda;
    // block 内 warp 的 id
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ int8_t s_a[BM * lda];
    __shared__ int8_t s_b[BK * ldb];
    __shared__ int s_c[BM * ldc];

    const int warp_col_id = warp_id % (BN / WN);
    const int warp_row_id = warp_id / (BN / WN);

    // lda = BK, ldb = BN, ldc = BN
    int8_t* a_ptr = s_a + warp_row_id * WM * lda;
    int8_t* b_ptr = s_b + warp_col_id * WN;

    using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::row_major>;
    using FragCType = wmma::fragment<wmma::accumulator, 16, 16, 16, int>;

    FragAType frag_a;
    FragBType frag_b;
    FragCType frag_acc;

    wmma::fill_fragment(frag_acc, 0);
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? clamp_int8(static_cast<int>(A[row_a * K + col_a]) - zero_point_A) : 0;
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? clamp_int8(static_cast<int>(B[row_b * N + col_b]) - zero_point_B) : 0;
        }
        __syncthreads();

        // compute block mma 
        wmma::load_matrix_sync(frag_a, a_ptr, lda);
        wmma::load_matrix_sync(frag_b, b_ptr, ldb);
        wmma::mma_sync(frag_acc, frag_a, frag_b, frag_acc);
        __syncthreads();
    }

    int* c_ptr = s_c + warp_row_id * WM * ldc + warp_col_id * WN;
    wmma::store_matrix_sync(c_ptr, frag_acc, ldc, wmma::mem_row_major);
    __syncthreads();

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            int acc = s_c[row * ldc + col];
            float scaled = (__int2float_rn(acc) * scale_A * scale_B / scale_C);
            C[row_c * N + col_c] = clamp_int8(__float2int_rn(scaled) + zero_point_C);
        }
    }
}

// A, B, C are device pointers
void int8_quantized_matmul_naive_cuda(
    const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 block(BM * BN);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    int8_quantized_matmul_kernel<BM, BN, BK><<<grid, block>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
} 


// A, B, C are device pointers
void int8_quantized_matmul_warp_wmma_cuda(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) 
{
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 16;
    constexpr int WM = 16;
    constexpr int WN = 16;

    dim3 block(BM * BN / (WM * WN) * 32);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    int8_quantized_matmul_warp_wmma_kernel<BM, BN, BK, WM, WN><<<grid, block>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


void int8_quantized_matmul_cuda(
    const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) 
{
    int8_quantized_matmul_warp_wmma_cuda(A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
}


torch::Tensor int8_quantized_matmul(
    const torch::Tensor& A,
    const torch::Tensor& B,
    float scale_A, float scale_B, float scale_C,
    int zero_point_A, int zero_point_B, int zero_point_C
) {
    TORCH_CHECK(A.ndimension() == 2, "Input A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "Input B must be 2D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Input A must be of type int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "Input B must be of type int8");
    TORCH_CHECK(A.size(1) == B.size(0), "Input tensors must have compatible shapes");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto options = A.options();
    torch::Tensor C = torch::empty({M, N}, options.dtype(torch::kInt8));

    int8_quantized_matmul_cuda(
        A.data_ptr<int8_t>(),
        B.data_ptr<int8_t>(),
        C.data_ptr<int8_t>(),
        M, N, K,
        scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C
    );
    return C;
}