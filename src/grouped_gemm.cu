#include "utils.cuh"

#include <torch/extension.h>



// -----------------------------------------------------------------------------
// Kernel config
// -----------------------------------------------------------------------------
namespace {
    constexpr int BM = 128;   // rows per output tile
    constexpr int BN = 128;   // cols per output tile
    constexpr int BK = 16;    // K-dimension tile
    constexpr int TM = 8;     // rows computed per thread (register sub-tile)
    constexpr int TN = 8;     // cols computed per thread
}


// -----------------------------------------------------------------------------
// Fused grouped GEMM kernel
//   blockDim = (BN/TN, BM/TM)
//   grid.x   = total_row_tiles (sum over groups of ceil(m_g / BM))
//   grid.y   = ceil(N / BN)
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_kernel(
    const float* __restrict__ A,            // (M_total, K) row-major
    const float* __restrict__ B,            // (G, K, N)   row-major
    const int*   __restrict__ group_offsets,// (G+1)
    const int*   __restrict__ row_tile_off, // (G+1) cumulative row-tile counts
    float*       __restrict__ C,            // (M_total, N) row-major
    int G, int K, int N)
{
    const int bx = blockIdx.y;              // column-tile index
    const int tile_x = blockIdx.x;          // global row-tile index (across all groups)

    // --- decode (g, local_by) from global row-tile index ---
    int g = 0;
    while (g < G && row_tile_off[g + 1] <= tile_x) ++g;
    const int s = group_offsets[g];
    const int e = group_offsets[g + 1];
    const int m_g = e - s;
    if (m_g == 0) return;                    // empty group -> no tiles
    const int local_by = tile_x - row_tile_off[g];
    const int row_base = s + local_by * BM;
    const int col_base = bx * BN;

    const float* Ag = A + (size_t)row_base * K;            // rows [row_base, +BM)
    const float* Bg = B + (size_t)g * K * N + col_base;    // col offset col_base
    float*       Cg = C + (size_t)row_base * N + col_base;

    const int m_tile = min(BM, m_g - local_by * BM);       // actual rows in this tile
    const int n_tile = min(BN, N - col_base);              // actual cols in this tile

    const int tx = threadIdx.x;          // 0..BN/TN-1  (col thread)
    const int ty = threadIdx.y;          // 0..BM/TM-1  (row thread)
    const int out_col = tx * TN;
    const int out_row = ty * TM;

    // double-buffered shared tiles
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0f;

    const int num_k = (K + BK - 1) / BK;

    // ---- load first K-tile (buffer 0) ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = out_row + i;
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk)
            As[0][r][kk] = (r < m_tile && kk < K) ? Ag[(size_t)r * K + kk] : 0.0f;
    }
    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = out_col + j;
            Bs[0][kk][c] = (c < n_tile && kk < K) ? Bg[(size_t)kk * N + c] : 0.0f;
        }
    }
    __syncthreads();

    // ---- main K-loop ----
    for (int kstep = 0; kstep < num_k; ++kstep) {
        const int buf = kstep & 1;
        const int nxt = buf ^ 1;

        // MMA with current buffer
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const float a = As[buf][out_row + i][kk];
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a * Bs[buf][kk][out_col + j];
            }
        }

        // load next K-tile into the other buffer (overlapped)
        if (kstep + 1 < num_k) {
            const int k0 = (kstep + 1) * BK;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                const int r = out_row + i;
                #pragma unroll
                for (int kk = 0; kk < BK; ++kk)
                    As[nxt][r][kk] = (r < m_tile && (k0 + kk) < K)
                                     ? Ag[(size_t)r * K + k0 + kk] : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < BK; ++kk) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    const int c = out_col + j;
                    Bs[nxt][kk][c] = (c < n_tile && (k0 + kk) < K)
                                     ? Bg[(size_t)(k0 + kk) * N + c] : 0.0f;
                }
            }
        }
        __syncthreads();
    }

    // ---- write back ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = out_row + i;
        if (r < m_tile) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int c = out_col + j;
                if (c < n_tile) Cg[(size_t)r * N + c] = acc[i][j];
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host dispatch
// -----------------------------------------------------------------------------
/**
 * @brief Perform grouped GEMM on the GPU.
 * @param d_A Pointer to the first input matrix A (M_total x K) in row-major order.
 * @param d_B Pointer to the second input matrix B (G x K x N) in row-major order.
 * @param d_group_offsets Pointer to the group offsets array (G+1).
 * @param d_C Pointer to the output matrix C (M_total x N) in row-major order.
 * @param G Number of groups.
 * @param M_total Total number of rows in matrix A.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param N Number of columns in matrix B and output matrix C.
 */
void grouped_gemm_cuda(
    const float* d_A, const float* d_B, const int* d_group_offsets,
    float* d_C, int G, int M_total, int K, int N)
{
    std::vector<int> h_off(G + 1);
    CHECK_CUDA_ERROR(cudaMemcpy(h_off.data(), d_group_offsets, (G + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // build cumulative row-tile counts on host (G <= 64, cheap)
    std::vector<int> h_rt(G + 1);
    h_rt[0] = 0;
    for (int g = 0; g < G; ++g) {
        int m_g = h_off[g + 1] - h_off[g];
        int rt = (m_g > 0) ? (m_g + BM - 1) / BM : 0;
        h_rt[g + 1] = h_rt[g] + rt;
    }
    int total_row_tiles = h_rt[G];
    int col_tiles = (N + BN - 1) / BN;

    int* d_rt;
    CHECK_CUDA_ERROR(cudaMalloc(&d_rt, (G + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rt, h_rt.data(), (G + 1) * sizeof(int), cudaMemcpyHostToDevice));

    if (total_row_tiles > 0) {
        dim3 grid(total_row_tiles, col_tiles);
        dim3 block(BN / TN, BM / TM);
        grouped_gemm_kernel<BM, BN, BK, TM, TN><<<grid, block, 0>>>(d_A, d_B, d_group_offsets, d_rt, d_C, G, K, N);
    }
    CHECK_CUDA_ERROR(cudaFree(d_rt));
}


torch::Tensor grouped_gemm(
    const torch::Tensor& A, 
    const torch::Tensor& B, 
    const torch::Tensor& group_offsets) 
{
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(group_offsets.dim() == 1, "group_offsets must be 1D");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(group_offsets.is_contiguous(), "group_offsets must be contiguous");

    int G = B.size(0);
    int M_total = A.size(0);
    int K = A.size(1);
    int N = B.size(2);

    auto C = torch::zeros({M_total, N}, A.options());

    grouped_gemm_cuda(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        group_offsets.data_ptr<int>(), 
        C.data_ptr<float>(), 
        G, M_total, K, N);

    return C;
}