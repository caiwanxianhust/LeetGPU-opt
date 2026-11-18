#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>


/**
 * @brief CUDA kernel for computing the backward pass of RMS normalization.
 * @param x Input tensor of shape (N, C).
 * @param weight Weight tensor of shape (C,).
 * @param dy Gradient of the loss with respect to the output of RMS normalization, shape (N, C).
 * @param dx Gradient of the loss with respect to the input x, shape (N, C).
 * @param dweight Gradient of the loss with respect to the weight, shape (C,).
 * @param N Number of samples (rows).
 * @param C Number of features (columns).
 * @param eps Small constant for numerical stability.
 */
static __global__ void rms_norm_backward_atomic_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ dy, 
    float* __restrict__ dx, float* __restrict__ dweight, int N, int C, float eps) 
{
    const float* x_row = x + blockIdx.x * C;
    const float* dy_row = dy + blockIdx.x * C;
    float* dx_row = dx + blockIdx.x * C;

    float r = 0.0f;
    float mean_gx = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float x_val = x_row[i];
        float dy_val = dy_row[i];
        float weight_val = weight[i];

        r += x_val * x_val;
        mean_gx += dy_val * weight_val * x_val;
    }
    // __syncthreads();

    r = block_all_reduce_sum(r);
    r = rsqrtf(r / (float)C + eps);
    mean_gx = block_all_reduce_sum(mean_gx);
    mean_gx /= (float)C;
    mean_gx *= r;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float x_val = x_row[i];
        float dy_val = dy_row[i];
        float weight_val = weight[i];

        dx_row[i] = r * (dy_val * weight_val - x_val * r * mean_gx);
        atomicAdd(&dweight[i], r * dy_val * x_val);
    }
}



// x, weight, dy, dx, dweight are device pointers
void rms_norm_backward_atomic(
    const float* x, const float* weight, const float* dy, float* dx, float* dweight, 
    int N, int C, float eps) 
{
    dim3 grid(N);
    dim3 block(256);
    rms_norm_backward_atomic_kernel<<<grid, block>>>(x, weight, dy, dx, dweight, N, C, eps);
}


// ---------- warp / block reduction (two values, broadcasts to ALL threads) ----------
static __device__ void warp_all_reduce_sum(float &a, float &b) {
    for (int offset = 16; offset > 0; offset /= 2) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

static __device__ void block_all_reduce_sum(float &a, float &b) {
    __shared__ float shared[32][2];   // up to 32 warps (<=1024 threads)
    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    warp_all_reduce_sum(a, b);        // stage 1: within-warp sum
    if (lane == 0) {
        shared[warp_id][0] = a;
        shared[warp_id][1] = b;
    }
    __syncthreads();

    a = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x][0] : 0.0f;
    b = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x][1] : 0.0f;
    warp_all_reduce_sum(a, b);        // stage 2: across-warp sum (only warp 0 holds the full sum)

    // broadcast the block-wide result to EVERY thread.
    // NOTE: the two-value version you pasted for LayerNorm dropped this step,
    // so threads in warp 1+ saw a=0,b=0 and dx was wrong for columns >= 32.
    if (threadIdx.x == 0) {
        shared[0][0] = a;
        shared[0][1] = b;
    }
    __syncthreads();
    a = shared[0][0];
    b = shared[0][1];
}

// ---------- kernel ----------
template <int VPT>
static __global__ void rms_norm_backward_atomic_opt_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ dy,
    float* __restrict__ dx, float* __restrict__ dweight,
    int N, int C, float eps)
{
    float acc[VPT];   // dweight partial: Σ over this block's rows of (r·dy·x)
#pragma unroll
    for (int k = 0; k < VPT; ++k) acc[k] = 0.f;

    for (int row_id = blockIdx.x; row_id < N; row_id += gridDim.x) {  // grid-stride over rows
        const float* x_row  = x  + (size_t)row_id * C;
        const float* dy_row = dy + (size_t)row_id * C;
        float* dx_row = dx + (size_t)row_id * C;

        float xv[VPT], dyv[VPT], wv[VPT];
        float sum_x2  = 0.0f;   // Σ x²   (per row)
        float sum_dxw = 0.0f;   // Σ dy·w·x (per row)

        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int col_id = threadIdx.x + blockDim.x * k;
            float x_val  = (col_id < C) ? x_row[col_id]  : 0.0f;
            float dy_val = (col_id < C) ? dy_row[col_id] : 0.0f;
            float w_val  = (col_id < C) ? weight[col_id] : 0.0f;
            xv[k] = x_val; dyv[k] = dy_val; wv[k] = w_val;
            sum_x2  += x_val * x_val;
            sum_dxw += dy_val * w_val * x_val;
        }

        // per-row block reduction -> r_i and mean_gx_i (broadcast to all threads)
        block_all_reduce_sum(sum_x2, sum_dxw);
        float r       = rsqrtf(sum_x2 / (float)C + eps);
        float mean_gx = (sum_dxw / (float)C) * r;   // = r·(1/C)·Σ dy·w·x

        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int col_id = threadIdx.x + blockDim.x * k;
            if (col_id < C) {
                float x_val = xv[k], dy_val = dyv[k], w_val = wv[k];
                // mean_gx already carries one r, so the single r in front is correct
                dx_row[col_id] = r * (dy_val * w_val - x_val * r * mean_gx);
                acc[k] += r * dy_val * x_val;   // dweight partial for this row
            }
        }
    }

    // single atomicAdd per block for dweight (nb·C atomic ops, not N·C)
    #pragma unroll
    for (int k = 0; k < VPT; ++k) {
        int col_id = threadIdx.x + blockDim.x * k;
        if (col_id < C) {
            atomicAdd(&dweight[col_id], acc[k]);
        }
    }
}

// ---------- host dispatch ----------
void rms_norm_backward_atomic_opt(
    const float* x, const float* weight, const float* dy, float* dx, float* dweight,
    int N, int C, float eps)
{
    if (C <= 0) return;
    if (N <= 0) {
        cudaMemsetAsync(dweight, 0, (size_t)C * sizeof(float));
        return;
    }

    int dev = 0, sms = 40, per = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);

    // atomicAdd accumulates -> dweight MUST be zero-initialized.
    // (test harness does NOT pre-zero outputs, so we memset here.)
    cudaMemsetAsync(dweight, 0, (size_t)C * sizeof(float));

    if (C <= 256) {
        constexpr int VPT = 1;  constexpr int BLOCK = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else if (C <= 512) {
        constexpr int VPT = 2;  constexpr int BLOCK = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else if (C <= 1024) {
        constexpr int VPT = 4;  constexpr int BLOCK = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else if (C <= 2048) {
        constexpr int VPT = 8;  constexpr int BLOCK = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else if (C <= 4096) {
        constexpr int VPT = 16; constexpr int BLOCK = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else if (C <= 8192) {
        constexpr int VPT = 16; constexpr int BLOCK = 512;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    } else {  // C is 16384
        constexpr int VPT = 32; constexpr int BLOCK = 512;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, rms_norm_backward_atomic_opt_kernel<VPT>, BLOCK, 0);
        if (per < 1) per = 1;
        int nb = sms * per; if (nb > N) nb = N;
        rms_norm_backward_atomic_opt_kernel<VPT><<<nb, BLOCK>>>(x, weight, dy, dx, dweight, N, C, eps);
    }
}








// x, weight, dy, dx, dweight are device pointers
void rms_norm_backward_cuda(
    const float* x, const float* weight, const float* dy, float* dx, float* dweight, 
    int N, int C, float eps) 
{
    // rms_norm_backward_atomic(x, weight, dy, dx, dweight, N, C, eps);
    rms_norm_backward_atomic_opt(x, weight, dy, dx, dweight, N, C, eps);
}



std::tuple<torch::Tensor, torch::Tensor> rms_norm_backward(
    const torch::Tensor& x, 
    const torch::Tensor& weight, 
    const torch::Tensor& dy, 
    float eps) 
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(dy.is_cuda(), "dy must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(dy.is_contiguous(), "dy must be contiguous");

    int N = x.size(0);
    int C = x.size(1);

    auto dx = torch::zeros_like(x);
    auto dweight = torch::zeros_like(weight);

    rms_norm_backward_cuda(
        x.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        dy.data_ptr<float>(), 
        dx.data_ptr<float>(), 
        dweight.data_ptr<float>(), 
        N, C, eps);

    return std::make_tuple(dx, dweight);
}