#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <math.h>
#include <torch/extension.h>


namespace others
{

// LayerNorm backward (mean / rstd given), FP32.
// Each block owns a strided set of rows (row = blockIdx.x + i*gridDim.x); the row's x and dy stay
// in registers (thread owns columns tid + NT*k). Per row: xhat = (x - mean)*rstd, g = dy*gamma,
// mean_g = mean(g), proj = mean(g*xhat) (one fused two-value block reduction),
// dx = rstd*((g - mean_g) - xhat*proj) with separate roundings like torch.
// dgamma / dbeta partials (sums of dy*xhat and dy over the block's rows) live in registers, are
// written to scratch[block][C] and summed over blocks in fixed order in double by a second kernel.
// No atomics: deterministic, outputs fully overwritten (zero dy -> exactly 0).

template <int NT, int VPT>
__global__ void __launch_bounds__(NT)
ln_bwd(const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ mean,
       const float* __restrict__ rstd, const float* __restrict__ dy, float* __restrict__ dx,
       float* __restrict__ pg, float* __restrict__ pb, int N, int C) {
    __shared__ float2 red[2][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float ag[VPT], ab[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) { ag[k] = 0.f; ab[k] = 0.f; }
    int buf = 0;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {  // block-uniform loop
        const float* xr = x + (size_t)row * C;
        const float* dr = dy + (size_t)row * C;
        const float mu = __ldg(&mean[row]);
        const float rs = __ldg(&rstd[row]);
        float xh[VPT], dv[VPT];
        float s1 = 0.f, s2 = 0.f;
#pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int c = tid + NT * k;
            bool ok = c < C;
            float xv = ok ? __ldg(&xr[c]) : 0.f;
            dv[k] = ok ? __ldg(&dr[c]) : 0.f;
            xh[k] = ok ? __fmul_rn(__fsub_rn(xv, mu), rs) : 0.f;
            float g = ok ? __fmul_rn(dv[k], __ldg(&gamma[c])) : 0.f;
            s1 += g;
            s2 = fmaf(g, xh[k], s2);
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            s2 += __shfl_xor_sync(0xffffffffu, s2, o);
        }
        if (lane == 0) red[buf][warp] = make_float2(s1, s2);
        __syncthreads();
        s1 = 0.f; s2 = 0.f;
#pragma unroll
        for (int i = 0; i < NT / 32; ++i) { float2 t = red[buf][i]; s1 += t.x; s2 += t.y; }
        buf ^= 1;  // next row writes the other buffer: no second barrier needed
        const float mean_g = s1 / (float)C;
        const float proj = s2 / (float)C;
        float* o = dx + (size_t)row * C;
#pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int c = tid + NT * k;
            if (c < C) {
                float g = __fmul_rn(dv[k], __ldg(&gamma[c]));
                o[c] = __fmul_rn(rs, __fsub_rn(__fsub_rn(g, mean_g), __fmul_rn(xh[k], proj)));
                ag[k] = fmaf(dv[k], xh[k], ag[k]);
                ab[k] += dv[k];
            }
        }
    }
    float* pgr = pg + (size_t)blockIdx.x * C;
    float* pbr = pb + (size_t)blockIdx.x * C;
#pragma unroll
    for (int k = 0; k < VPT; ++k) {
        int c = tid + NT * k;
        if (c < C) { pgr[c] = ag[k]; pbr[c] = ab[k]; }
    }
}

__global__ void col_reduce2(const float* __restrict__ pg, const float* __restrict__ pb, float* __restrict__ dg,
                            float* __restrict__ db, int nb, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double a = 0.0, b = 0.0;
    for (int i = 0; i < nb; ++i) {
        a += (double)pg[(size_t)i * C + c];
        b += (double)pb[(size_t)i * C + c];
    }
    dg[c] = (float)a;
    db[c] = (float)b;
}

static float* g_part = nullptr;
static size_t g_part_n = 0;

template <int NT, int VPT>
static void run_lnb(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy,
                    float* dx, float* dgamma, float* dbeta, int N, int C) {
    int dev = 0, sms = 40, per = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, ln_bwd<NT, VPT>, NT, 0);
    if (per < 1) per = 1;
    int nb = sms * per;
    if (nb > N) nb = N;
    size_t need = (size_t)nb * C * 2;
    if (need > g_part_n) {
        if (g_part) cudaFree(g_part);
        cudaMalloc(&g_part, need * sizeof(float));
        g_part_n = need;
    }
    float* pg = g_part;
    float* pb = g_part + (size_t)nb * C;
    ln_bwd<NT, VPT><<<nb, NT>>>(x, gamma, mean, rstd, dy, dx, pg, pb, N, C);
    col_reduce2<<<(C + 255) / 256, 256>>>(pg, pb, dgamma, dbeta, nb, C);
}

// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
void layer_norm_backward_twopass(const float* x, const float* gamma, const float* mean, const float* rstd, const float* dy,
                      float* dx, float* dgamma, float* dbeta, int N, int C) {
    if (C <= 0) return;
    if (N <= 0) {
        cudaMemsetAsync(dgamma, 0, (size_t)C * sizeof(float));
        cudaMemsetAsync(dbeta, 0, (size_t)C * sizeof(float));
        return;
    }
    if (C <= 256)        run_lnb<256, 1>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 512)   run_lnb<256, 2>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 1024)  run_lnb<256, 4>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 2048)  run_lnb<256, 8>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 4096)  run_lnb<256, 16>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else if (C <= 8192)  run_lnb<512, 16>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    else                 run_lnb<512, 32>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);  // C <= 16384
}


}   // namespace others





/**
 * @brief CUDA kernel for computing the backward pass of Layer normalization.
 * @param x Input tensor of shape (N, C).
 * @param gamma Scale tensor of shape (C,).
 * @param mean Mean tensor of shape (N,).
 * @param rstd Reciprocal of standard deviation tensor of shape (N,).
 * @param dy Gradient of the loss with respect to the output of Layer normalization, shape (N, C).
 * @param dx Gradient of the loss with respect to the input x, shape (N, C).
 * @param dgamma Gradient of the loss with respect to the scale parameter gamma, shape (C,).
 * @param dbeta Gradient of the loss with respect to the bias parameter beta, shape (C,).
 * @param N Number of samples (rows).
 * @param C Number of features (columns).
 */
static __global__ void layer_norm_backward_atomic_kernel(
    const float* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ mean, 
    const float* __restrict__ rstd, const float* __restrict__ dy, 
    float* __restrict__ dx, float* __restrict__ dgamma, float* __restrict__ dbeta, 
    int N, int C) 
{
    const float* x_row = x + blockIdx.x * C;
    const float* dy_row = dy + blockIdx.x * C;
    float* dx_row = dx + blockIdx.x * C;
    float mean_val = mean[blockIdx.x];
    float rstd_val = rstd[blockIdx.x];

    float g_mean = 0.0f;
    float gx_mean = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x){
        float g = dy_row[i] * gamma[i];
        g_mean += g;
        float x_hat = (x_row[i] - mean_val) * rstd_val;
        gx_mean += g * x_hat;

        atomicAdd(&dgamma[i], dy_row[i] * x_hat);
        atomicAdd(&dbeta[i], dy_row[i]);
    }

    g_mean = block_all_reduce_sum(g_mean);
    g_mean /= (float)C;
    gx_mean = block_all_reduce_sum(gx_mean);
    gx_mean /= (float)C;

    for (int i = threadIdx.x; i < C; i += blockDim.x){
        float g = dy_row[i] * gamma[i];
        dx_row[i] = rstd_val * (g - g_mean - (x_row[i] - mean_val) * rstd_val * gx_mean);
    }
}


// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
void layer_norm_backward_atomic(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) 
{
    
    dim3 block(256);
    dim3 grid(N);

    layer_norm_backward_atomic_kernel<<<grid, block>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
}


// reduce sum of two values across a warp
static __device__ void warp_all_reduce_sum(float &a, float &b) {
    for (int offset = 16; offset > 0; offset /= 2) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

static __device__ void block_all_reduce_sum(float &a, float &b) {
    __shared__ float shared[32][2]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    warp_all_reduce_sum(a, b);
    if (lane == 0) {
        shared[warp_id][0] = a;
        shared[warp_id][1] = b;
    }
    __syncthreads();

    a = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x][0] : 0.0f;
    b = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x][1] : 0.0f;
    warp_all_reduce_sum(a, b);

    if (threadIdx.x ==0 ){ 
        shared[0][0] = a; 
        shared[0][1] = b; 
    } 
    __syncthreads(); 
    a = shared[0][0]; 
    b = shared[0][1];
}



/**
 * @brief CUDA kernel for computing the backward pass of Layer normalization.
 * @param x Input tensor of shape (N, C).
 * @param gamma Scale tensor of shape (C,).
 * @param mean Mean tensor of shape (N,).
 * @param rstd Reciprocal of standard deviation tensor of shape (N,).
 * @param dy Gradient of the loss with respect to the output of Layer normalization, shape (N, C).
 * @param dx Gradient of the loss with respect to the input x, shape (N, C).
 * @param dgamma Gradient of the loss with respect to the scale parameter gamma, shape (C,).
 * @param dbeta Gradient of the loss with respect to the bias parameter beta, shape (C,).
 * @param N Number of samples (rows).
 * @param C Number of features (columns).
 * @param VPT Values per thread (number of columns processed by each thread).
 */
template <int VPT>
static __global__ void layer_norm_backward_atomic_opt_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ gamma, 
    const float* __restrict__ mean, 
    const float* __restrict__ rstd, 
    const float* __restrict__ dy, 
    float* __restrict__ dx, 
    float* __restrict__ dgamma, 
    float* __restrict__ dbeta, 
    int N, int C) 
{
    float acc_g[VPT], acc_b[VPT];
#pragma unroll
    for (int k = 0; k < VPT; ++k) { acc_g[k] = 0.f; acc_b[k] = 0.f; }

    for (int row_id = blockIdx.x; row_id < N; row_id += gridDim.x) {  // block-uniform loop
        const float* x_row = x + row_id * C;
        const float* dy_row = dy + row_id * C;
        float* dx_row = dx + row_id * C;
        float mean_val = mean[row_id];
        float rstd_val = rstd[row_id];

        float g[VPT];
        float x_hat[VPT];

        float g_mean = 0.0f;
        float gx_mean = 0.0f;

        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int col_id = threadIdx.x + blockDim.x * k;
            float dy_val = (col_id < C) ? dy_row[col_id] : 0.0f;
            g[k] = (col_id < C) ? dy_val * gamma[col_id] : 0.0f;
            x_hat[k] = (col_id < C) ? (x_row[col_id] - mean_val) : 0.0f;
            // thread accumulate g_mean and gx_mean for the current row
            g_mean += g[k];
            gx_mean += g[k] * x_hat[k];
            // thread accumulate dgamma and dbeta for the current row
            acc_g[k] += dy_val * x_hat[k];
            acc_b[k] += dy_val;
        }

        // block-level reduction to compute g_mean and gx_mean for the current row
        block_all_reduce_sum(g_mean, gx_mean);

        g_mean /= (float)C;
        gx_mean /= (float)C;

        #pragma unroll
        for (int k = 0; k < VPT; ++k) {
            int col_id = threadIdx.x + blockDim.x * k;
            if (col_id < C) {
                dx_row[col_id] = rstd_val * (g[k] - g_mean - x_hat[k] * gx_mean);
            }
        }
    }

    // After processing all rows, perform atomic adds for dgamma and dbeta
    #pragma unroll
    for (int k = 0; k < VPT; ++k) {
        int col_id = threadIdx.x + blockDim.x * k;
        if (col_id < C) {
            atomicAdd(&dgamma[col_id], acc_g[k]);
            atomicAdd(&dbeta[col_id], acc_b[k]);
        }
    }
}


// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
void layer_norm_backward_atomic_opt(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) 
{
    
    cudaMemset(dgamma, 0, (size_t)C * sizeof(float));
    cudaMemset(dbeta, 0, (size_t)C * sizeof(float));

    // Determine the number of streaming multiprocessors (sms) and the number of blocks per sm (per) for occupancy calculation
    int dev = 0, sms = 40, per = 1;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (C <= 256) {
        constexpr int VPT = 1;
        constexpr int BLOCK_SIZE = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<1><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else if (C <= 512) {
        constexpr int VPT = 2;
        constexpr int BLOCK_SIZE = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<2><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else if (C <= 1024) {
        constexpr int VPT = 4;
        constexpr int BLOCK_SIZE = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<4><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else if (C <= 2048) {
        constexpr int VPT = 8;
        constexpr int BLOCK_SIZE = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<8><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else if (C <= 4096) {
        constexpr int VPT = 16;
        constexpr int BLOCK_SIZE = 256;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<16><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else if (C <= 8192) {
        constexpr int VPT = 16;
        constexpr int BLOCK_SIZE = 512;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<16><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
    else {
        constexpr int VPT = 32;
        constexpr int BLOCK_SIZE = 512;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_norm_backward_atomic_opt_kernel<VPT>, BLOCK_SIZE, 0);
        if (per < 1) per = 1;
        int num_blocks = sms * per;
        if (num_blocks > N) num_blocks = N;
        layer_norm_backward_atomic_opt_kernel<32><<<num_blocks, BLOCK_SIZE>>>(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    }
}


// x, gamma, mean, rstd, dy, dx, dgamma, dbeta are device pointers
void layer_norm_backward_cuda(const float* x, const float* gamma, const float* mean, const float* rstd,
                      const float* dy, float* dx, float* dgamma, float* dbeta, int N, int C) 
{
    // You can choose which implementation to use here
    // others::layer_norm_backward_twopass(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    layer_norm_backward_atomic(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);
    // layer_norm_backward_atomic_opt(x, gamma, mean, rstd, dy, dx, dgamma, dbeta, N, C);

}



std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_backward(
    const torch::Tensor& x, 
    const torch::Tensor& gamma, 
    const torch::Tensor& mean, 
    const torch::Tensor& rstd, 
    const torch::Tensor& dy
)   {
    
    int N = x.size(0);
    int C = x.size(1);

    auto dx = torch::zeros_like(x);
    auto dgamma = torch::zeros_like(gamma);
    auto dbeta = torch::zeros_like(gamma);

    layer_norm_backward_cuda(
        x.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        mean.data_ptr<float>(), 
        rstd.data_ptr<float>(), 
        dy.data_ptr<float>(), 
        dx.data_ptr<float>(), 
        dgamma.data_ptr<float>(), 
        dbeta.data_ptr<float>(), 
        N, C
    );

    return std::make_tuple(dx, dgamma, dbeta);
}