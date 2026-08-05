#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cmath>
#include <cfloat>
#include <torch/extension.h>


/*
Implement a GPU program that computes the Fast Fourier Transform (FFT) of a complex-valued 1-D signal. Given an input signal array containing N complex numbers stored as interleaved real/imaginary pairs, compute the discrete Fourier transform and store the result in the spectrum array. The FFT converts a time-domain signal into its frequency-domain representation using the formula:
The FFT algorithm reduces the computational complexity from O(N²) to O(N log N) by exploiting symmetries in the twiddle factors.

Implementation Requirements
External libraries (cuFFT etc.) are not permitted
The solve function signature must remain unchanged
The final result must be stored in the spectrum array
The kernel must be entirely GPU-resident—no host-side FFT calls
Both input and output use interleaved real/imaginary layout: [real₀, imag₀, real₁, imag₁, ...]
Example 1:
Input:  N = 4
        signal = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        (represents: [1+0j, 0+0j, 0+0j, 0+0j])

Output: spectrum = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j, 1+0j, 1+0j])
Example 2:
Input:  N = 2
        signal = [1.0, 0.0, 1.0, 0.0]
        (represents: [1+0j, 1+0j])

Output: spectrum = [2.0, 0.0, 0.0, 0.0]
        (represents: [2+0j, 0+0j])
Constraints
1 ≤ N ≤ 262,144
All values are 32-bit floating point numbers
Absolute error ≤ 1e-3 and relative error ≤ 1e-3
Input and output arrays have length 2 × N
Performance is measured with N = 262,144
*/

static __host__ __device__ inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}



namespace dft_naive {

__global__ void dft_naive_kernel(const float* signal, float* spectrum, int N)
{ 
    const float2* signal_cplx = (const float2*) signal;
    float2* spectrum_cplx = (float2*) spectrum;

    float2 X_k = {0.0f, 0.0f}; // Initialize the output for this k-th thread

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float base_angle = -2.0f * (float)M_PI * (float)k / (float)N; 
    for (int n = 0; n < N; ++n) {
        float angle = base_angle * (float)n;
        float cos_angle;
        float sin_angle;
        sincosf(angle, &sin_angle, &cos_angle);
        float2 val = signal_cplx[n];

        X_k.x += val.x * cos_angle - val.y * sin_angle;
        X_k.y += val.x * sin_angle + val.y * cos_angle;
    }
    spectrum_cplx[k] = X_k;
}

void launch_dft_naive_kernel(const float* signal, float* spectrum, int N)
{
    dft_naive_kernel<<<div_up(N, 256), 256>>>(signal, spectrum, N);
}


}   // namespace dft_naive



namespace dft_reduce { 

__global__ void dft_reduce_kernel(const float* signal, float* spectrum, int N)
{ 
    const int k = blockIdx.x;
    float2 X_k = {0.0f, 0.0f}; // Initialize the output for this k-th thread
    float base_angle = -2.0f * (float)M_PI * (float)k / (float)N;

    for (int n = threadIdx.x; n < N; n += blockDim.x) { 
        float angle = base_angle * (float)n;
        float cos_angle;
        float sin_angle;
        sincosf(angle, &sin_angle, &cos_angle);
        float2 val = ((const float2*) signal)[n];

        X_k.x += val.x * cos_angle - val.y * sin_angle;
        X_k.y += val.x * sin_angle + val.y * cos_angle;
    }
    __syncthreads();
    X_k.x = block_all_reduce_sum(X_k.x);
    X_k.y = block_all_reduce_sum(X_k.y);

    if (threadIdx.x == 0) ((float2*)spectrum)[k] = X_k;
}


void launch_dft_reduce_kernel(const float* signal, float* spectrum, int N)
{
    dft_reduce_kernel<<<N, 256>>>(signal, spectrum, N);
}

}   // namespace dft_reduce


namespace fft_radix2 { 

/**
 * 检查整数 N 是否为 2 的幂
 */
static __host__ __device__ inline bool is_power_of_two(int N)
{
    return (N & (N - 1)) == 0;
}


/**
 * 位反转置换
 */
__device__ __forceinline__ int bit_reverse(int n, int log_N)
{
    int rev = 0;
    for (int i = 0; i < log_N; ++i) {
        rev = (rev << 1) | (n & 1);
        n >>= 1;
    }
    return rev;
}

__device__ __host__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __host__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}


__global__ void bit_reverse_permutation_kernel(const float* input, float* output, int N, int log_N) { 
    float2 * input_cplx = (float2*) input;
    float2 * output_cplx = (float2*) output;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int rev_idx = bit_reverse(idx, log_N);
    output_cplx[rev_idx] = input_cplx[idx];
}


__global__ void fft_stage_kernel(float* data, int N, int stage) {
    float2 * data_cplx = (float2*) data;

    int half = 1 << stage;
    int stride = half << 1;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int pos_in_group = idx & (stride - 1); // equivalent to idx % stride
    if (pos_in_group >= half) return;

    int j = idx + half;

    float2 val_i = data_cplx[idx];
    float2 val_j = data_cplx[j];

    float angle = (float)(- M_PI) * (float)pos_in_group / (float)half;
    float cos_angle;
    float sin_angle;
    sincosf(angle, &sin_angle, &cos_angle);

    float real = cos_angle * val_j.x - sin_angle * val_j.y;
    float imag = cos_angle * val_j.y + sin_angle * val_j.x;

    data_cplx[idx] = cadd(val_i, make_float2(real, imag));
    data_cplx[j] = csub(val_i, make_float2(real, imag));
}


void launch_fft_radix2_kernel(const float* signal, float* spectrum, int N) 
{
    constexpr int BLOCK_SIZE = 512;
    int logN = 0;
    for (int t = N; t > 1; t >>= 1) ++logN;
    bit_reverse_permutation_kernel<<<div_up(N, BLOCK_SIZE), BLOCK_SIZE>>>(signal, spectrum, N, logN);

    for (int stage = 0; stage < logN; ++stage) {
        fft_stage_kernel<<<div_up(N, BLOCK_SIZE), BLOCK_SIZE>>>(spectrum, N, stage);
    }
}

void fft_radix2_cuda(const float* signal, float* spectrum, int N)
{
    if (is_power_of_two(N)) {
        launch_fft_radix2_kernel(signal, spectrum, N);
    } else {
        // fallback to dft_reduce if N is not a power of two
        dft_naive::launch_dft_naive_kernel(signal, spectrum, N);
    }
}

}   // namespace fft_radix2



// signal and spectrum are device pointers
void fast_fft_cuda(const float* signal, float* spectrum, int N) 
{
    fft_radix2::fft_radix2_cuda(signal, spectrum, N);

}


torch::Tensor fast_fft(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(input.size(0) % 2 == 0, "Input size must be even");

    int N = input.size(0) / 2;
    torch::Tensor output = torch::empty_like(input);
    fast_fft_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}
