#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>

static constexpr int CHUNK_SIZE = 128;

// Head dim covered by the vectorized fast path. When D == D_SPECIALIZED one
// warp (32 lanes x float4) spans a whole row, so Q/K/V can be moved with 16-byte
// accesses and the D loop unrolls away. D == 128 makes every row stride
// (H * 128 floats) a multiple of 512 B, so the float4 loads are always aligned.
static constexpr int D_SPECIALIZED = 128;

static __global__ void flash_decoding_chunked_attention_kernel(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H, int D)
{
    // grid.x is the head and grid.y the chunk, so the blocks that are resident at
    // the same time cover all H heads of one row window: one row of K/V is
    // H * D * 4 contiguous bytes, and the blocks read it as a whole instead of
    // each one pulling a lone D * 4 byte slice out of a row that is H times wider.
    // That keeps the DRAM pages of a row fully used while they are open.
    const int chunk_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int query_idx = blockIdx.z;
    const int start_chunk = chunk_idx * CHUNK_SIZE;
    const int eff_chunk = min(CHUNK_SIZE, S - start_chunk);
    const int num_chunks = div_up(S, CHUNK_SIZE);

    const float* Q_ptr = Q + (query_idx * H + head_idx) * D;
    const float* K_ptr = K + (start_chunk * H + head_idx) * D;
    const float* V_ptr = V + (start_chunk * H + head_idx) * D;
    float * output_ptr = O_chunk + ((query_idx * H + head_idx) * num_chunks + chunk_idx) * D;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));

    extern __shared__ float s_buf[];
    float* s_Q = s_buf;
    // load Q into shared memory
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        s_Q[i] = Q_ptr[i];
    }
    __syncthreads();

    __shared__ float s_qk[CHUNK_SIZE];

    float max_score = -1e20f;
    float sum_exp_score = 0.0f;

    // comnpute qk, each warp processed one k row
    for (int row_id = warp_id; row_id < eff_chunk; row_id += (blockDim.x >> 5)) {
        float qk = 0.0f;
        for (int i = lane_id; i < D; i += 32) {
            qk += s_Q[i] * K_ptr[row_id * H * D + i];
        }
        qk = warp_all_reduce_sum(qk);
        if (lane_id == 0) {
            s_qk[row_id] = qk * softmax_scale;
        }
        __syncwarp();
    }
    __syncthreads();

    // compute max score
    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        max_score = max(max_score, s_qk[i]);
    }
    max_score = block_all_reduce_max(max_score);

    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        float exp_score = expf(s_qk[i] - max_score);
        s_qk[i] = exp_score;
        sum_exp_score += exp_score;
    }
    sum_exp_score = block_all_reduce_sum(sum_exp_score);

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.0f;
        for (int row_id = 0; row_id < eff_chunk; ++row_id) {
            float weight = s_qk[row_id] / sum_exp_score;
            float v = V_ptr[row_id * H * D + i];
            acc += weight * v;
        }
        output_ptr[i] = acc;
    }

    // LSE
    if (threadIdx.x == 0) {
        const int lse_offset = query_idx * H * num_chunks + head_idx * num_chunks + chunk_idx;
        lse[lse_offset] = max_score + logf(sum_exp_score);
    }
}


// D = 128 fast path. blockDim.x must be 128 (4 warps); one warp covers a whole
// 128-wide row with one float4 per lane, so:
//   * Q is loaded once per warp into registers instead of shared memory,
//   * K/V moves with 16-byte loads (4x fewer L1TEX requests than the generic path),
//   * the PV reduction is split over the 4 warps and combined in shared memory.
static __global__ void flash_decoding_chunked_attention_kernel_d128(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H)
{
    constexpr int D = D_SPECIALIZED;
    constexpr int NUM_WARPS = 4;
    constexpr int ROWS_PER_WARP = CHUNK_SIZE / NUM_WARPS;   // 32
    constexpr int COLS_PER_LANE = D / 32;                   // 4

    // Head-major block order, same reason as the generic kernel above.
    const int chunk_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int query_idx = blockIdx.z;
    const int start_chunk = chunk_idx * CHUNK_SIZE;
    const int eff_chunk = min(CHUNK_SIZE, S - start_chunk);
    const int num_chunks = div_up(S, CHUNK_SIZE);

    const float* Q_ptr = Q + (query_idx * H + head_idx) * D;
    const float* K_ptr = K + (start_chunk * H + head_idx) * D;
    const float* V_ptr = V + (start_chunk * H + head_idx) * D;
    float* output_ptr = O_chunk + ((query_idx * H + head_idx) * num_chunks + chunk_idx) * D;

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int col = lane_id * COLS_PER_LANE;

    const float softmax_scale = 1.0f / sqrtf(static_cast<float>(D));

    // Q row lives in 4 registers for the rest of the kernel.
    const float4 q = *reinterpret_cast<const float4*>(Q_ptr + col);

    __shared__ float s_qk[CHUNK_SIZE];
    __shared__ float s_part[NUM_WARPS][D];

    // QK: warp w handles rows w, w + NUM_WARPS, ... (one float4 of K per row).
    // Rows are taken four at a time: the four 16-byte loads are issued before the
    // first dot product consumes them, so four loads stay in flight per lane.
    // On parts where the kernel is latency- rather than bandwidth-bound (few SMs
    // relative to DRAM bandwidth) that is what keeps the memory pipeline full.
    const int64_t row_stride = static_cast<int64_t>(H) * D;
    const int64_t quad_stride = static_cast<int64_t>(NUM_WARPS) * row_stride;
    int row_id = warp_id;
    for (; row_id + 3 * NUM_WARPS < eff_chunk; row_id += 4 * NUM_WARPS) {
        const int64_t r = static_cast<int64_t>(row_id) * row_stride + col;
        const float4 k0 = __ldg(reinterpret_cast<const float4*>(K_ptr + r));
        const float4 k1 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + quad_stride));
        const float4 k2 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + 2 * quad_stride));
        const float4 k3 = __ldg(reinterpret_cast<const float4*>(K_ptr + r + 3 * quad_stride));
        float qk0 = q.x * k0.x + q.y * k0.y + q.z * k0.z + q.w * k0.w;
        float qk1 = q.x * k1.x + q.y * k1.y + q.z * k1.z + q.w * k1.w;
        float qk2 = q.x * k2.x + q.y * k2.y + q.z * k2.z + q.w * k2.w;
        float qk3 = q.x * k3.x + q.y * k3.y + q.z * k3.z + q.w * k3.w;
        qk0 = warp_all_reduce_sum(qk0);
        qk1 = warp_all_reduce_sum(qk1);
        qk2 = warp_all_reduce_sum(qk2);
        qk3 = warp_all_reduce_sum(qk3);
        if (lane_id == 0) {
            s_qk[row_id] = qk0 * softmax_scale;
            s_qk[row_id + NUM_WARPS] = qk1 * softmax_scale;
            s_qk[row_id + 2 * NUM_WARPS] = qk2 * softmax_scale;
            s_qk[row_id + 3 * NUM_WARPS] = qk3 * softmax_scale;
        }
    }
    for (; row_id < eff_chunk; row_id += NUM_WARPS) {
        const float4 k = __ldg(reinterpret_cast<const float4*>(
            K_ptr + static_cast<int64_t>(row_id) * row_stride + col));
        float qk = q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
        qk = warp_all_reduce_sum(qk);
        if (lane_id == 0) {
            s_qk[row_id] = qk * softmax_scale;
        }
    }
    __syncthreads();

    float max_score = -1e20f;
    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        max_score = max(max_score, s_qk[i]);
    }
    max_score = block_all_reduce_max(max_score);

    float sum_exp_score = 0.0f;
    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        float exp_score = expf(s_qk[i] - max_score);
        s_qk[i] = exp_score;
        sum_exp_score += exp_score;
    }
    sum_exp_score = block_all_reduce_sum(sum_exp_score);

    // Normalise once instead of doing a division per (row, column) pair.
    const float inv_sum = 1.0f / sum_exp_score;
    for (int i = threadIdx.x; i < eff_chunk; i += blockDim.x) {
        s_qk[i] *= inv_sum;
    }
    __syncthreads();

    // PV: warp w owns the contiguous row range [w * ROWS_PER_WARP, ...), so the
    // per-warp partials are combined once in shared memory at the end. Four rows
    // are prefetched per iteration (same reasoning as the QK loop above).
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int row_begin = warp_id * ROWS_PER_WARP;
    const int row_end = min(eff_chunk, row_begin + ROWS_PER_WARP);
    row_id = row_begin;
    for (; row_id + 3 < row_end; row_id += 4) {
        const int64_t r = static_cast<int64_t>(row_id) * row_stride + col;
        const float4 v0 = __ldg(reinterpret_cast<const float4*>(V_ptr + r));
        const float4 v1 = __ldg(reinterpret_cast<const float4*>(V_ptr + r + row_stride));
        const float4 v2 = __ldg(reinterpret_cast<const float4*>(V_ptr + r + 2 * row_stride));
        const float4 v3 = __ldg(reinterpret_cast<const float4*>(V_ptr + r + 3 * row_stride));
        const float w0 = s_qk[row_id + 0];
        const float w1 = s_qk[row_id + 1];
        const float w2 = s_qk[row_id + 2];
        const float w3 = s_qk[row_id + 3];
        acc.x += w0 * v0.x;
        acc.x += w1 * v1.x;
        acc.x += w2 * v2.x;
        acc.x += w3 * v3.x;
        acc.y += w0 * v0.y;
        acc.y += w1 * v1.y;
        acc.y += w2 * v2.y;
        acc.y += w3 * v3.y;
        acc.z += w0 * v0.z;
        acc.z += w1 * v1.z;
        acc.z += w2 * v2.z;
        acc.z += w3 * v3.z;
        acc.w += w0 * v0.w;
        acc.w += w1 * v1.w;
        acc.w += w2 * v2.w;
        acc.w += w3 * v3.w;
    }
    for (; row_id < row_end; ++row_id) {
        const float weight = s_qk[row_id];
        const float4 v = __ldg(reinterpret_cast<const float4*>(
            V_ptr + static_cast<int64_t>(row_id) * row_stride + col));
        acc.x += weight * v.x;
        acc.y += weight * v.y;
        acc.z += weight * v.z;
        acc.w += weight * v.w;
    }

    s_part[warp_id][col + 0] = acc.x;
    s_part[warp_id][col + 1] = acc.y;
    s_part[warp_id][col + 2] = acc.z;
    s_part[warp_id][col + 3] = acc.w;
    __syncthreads();

    // Warp 0 writes the 128 outputs with one float4 per lane.
    if (threadIdx.x < 32) {
        float4 o;
        o.x = s_part[0][col + 0] + s_part[1][col + 0] + s_part[2][col + 0] + s_part[3][col + 0];
        o.y = s_part[0][col + 1] + s_part[1][col + 1] + s_part[2][col + 1] + s_part[3][col + 1];
        o.z = s_part[0][col + 2] + s_part[1][col + 2] + s_part[2][col + 2] + s_part[3][col + 2];
        o.w = s_part[0][col + 3] + s_part[1][col + 3] + s_part[2][col + 3] + s_part[3][col + 3];
        *reinterpret_cast<float4*>(output_ptr + col) = o;
    }

    // LSE
    if (threadIdx.x == 0) {
        const int lse_offset = query_idx * H * num_chunks + head_idx * num_chunks + chunk_idx;
        lse[lse_offset] = max_score + logf(sum_exp_score);
    }
}


static __global__ void flash_decoding_reduce_attention_kernel(
    float* __restrict__ output,
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H, int D
) {
    const int query_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_chunks = div_up(S, CHUNK_SIZE);

    float* O_ptr = output + (query_idx * H + head_idx) * D;
    float* O_chunk_ptr = O_chunk + (query_idx * H + head_idx) * num_chunks * D;
    float* lse_ptr = lse + (query_idx * H + head_idx) * num_chunks;

    extern __shared__ float s_buf[];
    float* s_lse = s_buf;               // [num_chunks,]

    float max_lse = -1e20f;
    // Phase 0: lse 整块搬进 shared（只从 global 读这一次）
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
    {
        float lse_val = lse_ptr[i];
        max_lse = max(max_lse, lse_val);
        s_lse[i] = lse_val;
    }
    __syncthreads();

    max_lse = block_all_reduce_max(max_lse);

    float deno = 0.0f;
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
    {
        deno += expf(s_lse[i] - max_lse);
    }
    deno = block_all_reduce_sum(deno);

    float global_lse = max_lse + logf(deno);

    // lse_s 原地改写成权重
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x)
        s_lse[i] = expf(s_lse[i] - global_lse);
    __syncthreads();

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float acc = 0.0f;
        for (int row_id = 0; row_id < num_chunks; row_id++) {
            float w = s_lse[row_id];
            acc += O_chunk_ptr[row_id * D + i] * w;
        }
        O_ptr[i] = acc;
    }
}


// D = 128 fast path for the combine pass. blockDim.x must be 128 (4 warps).
// The weight table is built once per block (the generic version recomputes
// expf(lse_i - global_lse) for every thread and every chunk), and the partial
// outputs are streamed with float4 loads split over the 4 warps.
static __global__ void flash_decoding_reduce_attention_kernel_d128(
    float* __restrict__ output,
    float* __restrict__ O_chunk,                // [T, H, num_chunks, D]
    float* __restrict__ lse,                    // [T, H, num_chunks]
    int T, int S, int H
) {
    constexpr int D = D_SPECIALIZED;
    constexpr int NUM_WARPS = 4;
    constexpr int COLS_PER_LANE = D / 32;                   // 4

    const int query_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_chunks = div_up(S, CHUNK_SIZE);

    float* O_ptr = output + (query_idx * H + head_idx) * D;
    float* O_chunk_ptr = O_chunk + (query_idx * H + head_idx) * num_chunks * D;
    float* lse_ptr = lse + (query_idx * H + head_idx) * num_chunks;

    extern __shared__ float s_buf[];
    float* s_w = s_buf;                         // [num_chunks,] normalised weights
    __shared__ float s_part[NUM_WARPS][D];

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int col = lane_id * COLS_PER_LANE;

    float max_lse = -1e20f;
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        float lse_val = __ldg(lse_ptr + i);
        max_lse = max(max_lse, lse_val);
        s_w[i] = lse_val;
    }
    __syncthreads();

    max_lse = block_all_reduce_max(max_lse);

    float deno = 0.0f;
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        deno += expf(s_w[i] - max_lse);
    }
    deno = block_all_reduce_sum(deno);

    const float global_lse = max_lse + logf(deno);

    // Build the weight table once, then every warp just reads it.
    for (int i = threadIdx.x; i < num_chunks; i += blockDim.x) {
        s_w[i] = expf(s_w[i] - global_lse);
    }
    __syncthreads();

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const int64_t o_row_stride = static_cast<int64_t>(NUM_WARPS) * D;
    int ci = warp_id;
    for (; ci + 3 * NUM_WARPS < num_chunks; ci += 4 * NUM_WARPS) {
        const int64_t r = static_cast<int64_t>(ci) * D + col;
        const float4 o0 = __ldg(reinterpret_cast<const float4*>(O_chunk_ptr + r));
        const float4 o1 = __ldg(reinterpret_cast<const float4*>(O_chunk_ptr + r + o_row_stride));
        const float4 o2 = __ldg(reinterpret_cast<const float4*>(O_chunk_ptr + r + 2 * o_row_stride));
        const float4 o3 = __ldg(reinterpret_cast<const float4*>(O_chunk_ptr + r + 3 * o_row_stride));
        const float w0 = s_w[ci];
        const float w1 = s_w[ci + NUM_WARPS];
        const float w2 = s_w[ci + 2 * NUM_WARPS];
        const float w3 = s_w[ci + 3 * NUM_WARPS];
        acc.x += w0 * o0.x;
        acc.x += w1 * o1.x;
        acc.x += w2 * o2.x;
        acc.x += w3 * o3.x;
        acc.y += w0 * o0.y;
        acc.y += w1 * o1.y;
        acc.y += w2 * o2.y;
        acc.y += w3 * o3.y;
        acc.z += w0 * o0.z;
        acc.z += w1 * o1.z;
        acc.z += w2 * o2.z;
        acc.z += w3 * o3.z;
        acc.w += w0 * o0.w;
        acc.w += w1 * o1.w;
        acc.w += w2 * o2.w;
        acc.w += w3 * o3.w;
    }
    for (; ci < num_chunks; ci += NUM_WARPS) {
        const float weight = s_w[ci];
        const float4 o = __ldg(reinterpret_cast<const float4*>(
            O_chunk_ptr + static_cast<int64_t>(ci) * D + col));
        acc.x += weight * o.x;
        acc.y += weight * o.y;
        acc.z += weight * o.z;
        acc.w += weight * o.w;
    }

    s_part[warp_id][col + 0] = acc.x;
    s_part[warp_id][col + 1] = acc.y;
    s_part[warp_id][col + 2] = acc.z;
    s_part[warp_id][col + 3] = acc.w;
    __syncthreads();

    if (threadIdx.x < 32) {
        float4 o;
        o.x = s_part[0][col + 0] + s_part[1][col + 0] + s_part[2][col + 0] + s_part[3][col + 0];
        o.y = s_part[0][col + 1] + s_part[1][col + 1] + s_part[2][col + 1] + s_part[3][col + 1];
        o.z = s_part[0][col + 2] + s_part[1][col + 2] + s_part[2][col + 2] + s_part[3][col + 2];
        o.w = s_part[0][col + 3] + s_part[1][col + 3] + s_part[2][col + 3] + s_part[3][col + 3];
        *reinterpret_cast<float4*>(O_ptr + col) = o;
    }
}





static void flash_decoding_chunked_attention(
    const float* Q, const float* K, const float* V, float* output,
    float* d_O_chunk, float* d_lse,
    int T, int S, int H, int D)
{
    const int num_chunks = div_up(S, CHUNK_SIZE);
    dim3 block(128);

    if (D == D_SPECIALIZED) {
        dim3 grid(H, num_chunks, T);   // x = head, y = chunk
        flash_decoding_chunked_attention_kernel_d128<<<grid, block>>>(
            Q, K, V, d_O_chunk, d_lse, T, S, H);
        CHECK_CUDA_ERROR(cudaGetLastError());

        dim3 reduce_grid(H, T);
        flash_decoding_reduce_attention_kernel_d128<<<reduce_grid, block, num_chunks * sizeof(float)>>>(
            output, d_O_chunk, d_lse, T, S, H);
        CHECK_CUDA_ERROR(cudaGetLastError());
        return;
    }

    dim3 grid(H, num_chunks, T);   // x = head, y = chunk
    flash_decoding_chunked_attention_kernel<<<grid, block, D * sizeof(float)>>>(
        Q, K, V, d_O_chunk, d_lse, T, S, H, D);
    CHECK_CUDA_ERROR(cudaGetLastError());

    grid = dim3(H, T);
    block = dim3(128);
    flash_decoding_reduce_attention_kernel<<<grid, block, num_chunks * sizeof(float)>>>(
        output, d_O_chunk, d_lse, T, S, H, D);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


/**
 * @brief Flash decoding attention CUDA kernel launcher.
 * @param Q [T, H, D] Pointer to the query tensor on the device. 
 * @param K [S, H, D] Pointer to the key tensor on the device. 
 * @param V [S, H, D] Pointer to the value tensor on the device. 
 * @param output [T, H, D] Pointer to the output tensor on the device. 
 * @param T Length of the query sequence.
 * @param S Length of the key/value sequence.
 * @param H Number of heads.
 * @param D Dimension of each head.
 * @details Constraints
1 ≤ T ≤ 16
1 ≤ S ≤ 131,072
1 ≤ H ≤ 64
1 ≤ D ≤ 256
Input values are in [-6, 6]
Performance is measured with T = 1, S = 65,536, H = 32, D = 128

The two-pass workspace (per-chunk partial outputs + per-chunk log-sum-exp) is
kept in a file-static device buffer and only reallocated when a call needs more
of it: a cudaMalloc/cudaFree pair costs several hundred microseconds (cudaFree
additionally waits for the device to go idle), which would otherwise be paid on
every invocation.
 */
// Q, K, V, output are device pointers
void flash_decoding_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, 
    int T, int S, int H, int D) 
{
    const int num_chunks = div_up(S, CHUNK_SIZE);
    const int64_t O_chunk_elems = static_cast<int64_t>(T) * H * num_chunks * D;
    const int64_t lse_elems = static_cast<int64_t>(T) * H * num_chunks;
    const int64_t scratch_elems = O_chunk_elems + lse_elems;

    static float* g_scratch = nullptr;
    static int64_t g_scratch_elems = 0;
    if (scratch_elems > g_scratch_elems) {
        if (g_scratch != nullptr) {
            CHECK_CUDA_ERROR(cudaFree(g_scratch));
        }
        CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&g_scratch),
                                    static_cast<size_t>(scratch_elems) * sizeof(float)));
        g_scratch_elems = scratch_elems;
    }
    float* d_O_chunk = g_scratch;
    float* d_lse = d_O_chunk + O_chunk_elems;

    flash_decoding_chunked_attention(Q, K, V, output, d_O_chunk, d_lse, T, S, H, D);
}



torch::Tensor flash_decoding_attention(
    const torch::Tensor & Q,
    const torch::Tensor & K,
    const torch::Tensor & V
) {
    TORCH_CHECK(Q.dim() == 3, "Q must be a 3D tensor [T, H, D]");
    TORCH_CHECK(K.dim() == 3, "K must be a 3D tensor [S, H, D]");
    TORCH_CHECK(V.dim() == 3, "V must be a 3D tensor [S, H, D]");
    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(K.sizes().equals(V.sizes()), "K and V must have the same shape [S, H, D]");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                "Q, K and V must live on the same device");
    TORCH_CHECK(Q.size(0) >= 1 && K.size(0) >= 1, "T and S must be at least 1");
    TORCH_CHECK(Q.size(1) == K.size(1),
                "Q and K must agree on the number of heads H");
    TORCH_CHECK(Q.size(2) == K.size(2),
                "Q and K must agree on the head dimension D");
    TORCH_CHECK(Q.size(2) >= 1, "D must be at least 1");

    const int T = Q.size(0);
    const int S = K.size(0);
    const int H = K.size(1);
    const int D = K.size(2);
    torch::Tensor output = torch::empty_like(Q);

    flash_decoding_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        T, S, H, D
    );
    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}