#include "utils.cuh"

#include <cufft.h>

/*
Compute the 2D Discrete Fourier Transform (2D DFT) of a complex-valued signal stored on the GPU. Given a 2D complex input signal of shape M × N, compute its 2D DFT spectrum using the row-column decomposition: apply a 1D DFT along each row, then a 1D DFT along each column of the result. All values are 32-bit floating point.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in spectrum
The input and output are stored as 1D arrays of interleaved real and imaginary parts in row-major order: element x[m, n] has its real part at index 2*(m*N + n) and imaginary part at index 2*(m*N + n) + 1

Constraints
1 ≤ M, N ≤ 4096
Signal values are 32-bit floating point (real and imaginary parts)
Performance is measured with M = 2,048, N = 2,048
*/


namespace reBourne87 {

constexpr float PI_F = 3.14159265358979323846f;
constexpr double PI_D = 3.14159265358979323846264338327950288;

static float2* g_buf1 = nullptr;
static float2* g_buf2 = nullptr;
static size_t g_capacity_complex = 0;

__device__ __forceinline__ float2 cadd(const float2 a, const float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub(const float2 a, const float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 cmul(const float2 a, const float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ unsigned bit_reverse(unsigned x, int logn) {
    unsigned r = 0;
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        if (i < logn) {
            r = (r << 1) | (x & 1u);
            x >>= 1u;
        }
    }
    return r;
}

__global__ void bit_reverse_copy_kernel(const float2* __restrict__ src,
                                        float2* __restrict__ dst,
                                        int batch, int len, int logn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * len;
    if (idx >= total) return;

    int seq = idx / len;
    int j = idx - seq * len;
    unsigned rev = bit_reverse((unsigned)j, logn);
    dst[idx] = src[seq * len + (int)rev];
}

__global__ void fft_stage_kernel(float2* __restrict__ data,
                                 int batch, int len, int span) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = span >> 1;
    int total_pairs = batch * (len >> 1);
    if (idx >= total_pairs) return;

    int seq = idx / (len >> 1);
    int p = idx - seq * (len >> 1);

    int group = p / half;
    int j = p - group * half;

    int base = seq * len + group * span + j;
    int mate = base + half;

    float angle = -2.0f * PI_F * (float)j / (float)span;
    float s, c;
    s = sinf(angle);
    c = cosf(angle);
    float2 w = make_float2(c, s);

    float2 u = data[base];
    float2 v = data[mate];
    float2 t = cmul(w, v);

    data[base] = cadd(u, t);
    data[mate] = csub(u, t);
}

// High-accuracy fallback for non-power-of-two lengths.
__global__ void dft_batched_kernel(const float2* __restrict__ src,
                                   float2* __restrict__ dst,
                                   int batch, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * len;
    if (idx >= total) return;

    int seq = idx / len;
    int k = idx - seq * len;

    const float2* row = src + seq * len;

    double acc_r = 0.0;
    double acc_i = 0.0;

    for (int n = 0; n < len; ++n) {
        double angle = -2.0 * PI_D * (double)(k * n) / (double)len;
        double s = sin(angle);
        double c = cos(angle);

        double xr = (double)row[n].x;
        double xi = (double)row[n].y;

        acc_r += xr * c - xi * s;
        acc_i += xr * s + xi * c;
    }

    dst[idx] = make_float2((float)acc_r, (float)acc_i);
}

template <int TILE, int BLOCK_ROWS>
__global__ void transpose_kernel(const float2* __restrict__ in,
                                 float2* __restrict__ out,
                                 int rows, int cols) {
    __shared__ float2 tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

#pragma unroll
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = in[(y + i) * cols + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

#pragma unroll
    for (int i = 0; i < TILE; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            out[(y + i) * rows + x] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

bool ensure_buffers(size_t complex_count) {
    if (complex_count <= g_capacity_complex) return true;

    if (g_buf1) cudaFree(g_buf1);
    if (g_buf2) cudaFree(g_buf2);
    g_buf1 = nullptr;
    g_buf2 = nullptr;
    g_capacity_complex = 0;

    size_t bytes = complex_count * sizeof(float2);
    if (cudaMalloc(&g_buf1, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&g_buf2, bytes) != cudaSuccess) {
        cudaFree(g_buf1);
        g_buf1 = nullptr;
        return false;
    }

    g_capacity_complex = complex_count;
    return true;
}

inline bool is_power_of_two(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

inline int ilog2_exact(int x) {
    int r = 0;
    while ((1 << r) < x) ++r;
    return r;
}

void launch_fft_rows_pow2(const float2* src, float2* dst, int batch, int len) {
    const int threads = 256;
    const int total = batch * len;
    const int logn = ilog2_exact(len);

    bit_reverse_copy_kernel<<<(total + threads - 1) / threads, threads>>>(src, dst, batch, len, logn);

    for (int span = 2; span <= len; span <<= 1) {
        int total_pairs = batch * (len >> 1);
        fft_stage_kernel<<<(total_pairs + threads - 1) / threads, threads>>>(dst, batch, len, span);
    }
}

void launch_dft_rows(const float2* src, float2* dst, int batch, int len) {
    const int threads = 256;
    const int total = batch * len;
    dft_batched_kernel<<<(total + threads - 1) / threads, threads>>>(src, dst, batch, len);
}

void launch_transform_rows(const float2* src, float2* dst, int batch, int len) {
    if (is_power_of_two(len)) {
        launch_fft_rows_pow2(src, dst, batch, len);
    } else {
        launch_dft_rows(src, dst, batch, len);
    }
}

void launch_transpose(const float2* in, float2* out, int rows, int cols) {
    constexpr int TILE = 32;
    constexpr int BLOCK_ROWS = 8;
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    transpose_kernel<TILE, BLOCK_ROWS><<<grid, block>>>(in, out, rows, cols);
}

// signal, spectrum are device pointers
void fft_2D_reBourne87_cuda(const float* signal, float* spectrum, int M, int N) {
    size_t total_complex = static_cast<size_t>(M) * static_cast<size_t>(N);
    if (!ensure_buffers(total_complex)) return;

    const float2* signal_c = reinterpret_cast<const float2*>(signal);
    float2* spectrum_c = reinterpret_cast<float2*>(spectrum);

    // Rows
    launch_transform_rows(signal_c, g_buf1, M, N);

    // Transpose
    launch_transpose(g_buf1, g_buf2, M, N);

    // Columns (as rows on transposed matrix)
    launch_transform_rows(g_buf2, g_buf1, N, M);

    // Transpose back
    launch_transpose(g_buf1, spectrum_c, N, M);
}


}  // namespace reBourne87


namespace cufft_api {

void fft_2D_cufft_cuda(const float* signal, float* spectrum, int M, int N) {

    cufftComplex* signal_c = (cufftComplex*)signal;
    cufftComplex* spectrum_c = (cufftComplex*)spectrum;

    /*
    cufftResult cufftPlanMany(
        cufftHandle *plan,
        int rank,         // FFT维度(1/2/3)
        int *n,           // 每个维度的FFT点数
        int *inembed,     // 输入数据各维度实际尺寸
        int istride,      // 输入数据采样步长
        int idist,        // 输入数据批次间隔
        int *onembed,     // 输出数据各维度实际尺寸
        int ostride,      // 输出数据采样步长
        int odist,        // 输出数据批次间隔
        cufftType type,   // 变换类型(C2C/R2C等)
        int batch         // 批量处理次数
    );
    */

    /*
    cufftHandle fft_2D_plan;
    int rank = 2;
    int n[2] = {N, M};
    int inembed[2] = {N, M};
    int istride = 1;
    int idist = N * M;
    int onembed[2] = {N, M};
    int ostride = 1;
    int odist = N * M;
    cufftType type = CUFFT_C2C;
    int batch = 1;

    CHECK_CUFFT_STATUS(cufftPlanMany(&fft_2D_plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch));
    */

    /*
    函数：
        cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type)
 
    功能：
        根据指定的信号大小和数据类型创建2D FFT计划配置
    
    输入参数：
        plan：        cufftHandle 指针
        nx：          可以视为一个矩阵的列
        ny:           可以视为一个矩阵的行
        type：        用于执行傅里叶变换的数据类型，比如：CUFFT_C2C,CUFFT_C2R等
    
    输出参数：
        plan：二维傅里叶变换句柄
    
    返回值：
        CUFFT_SUCCESS：           成功创建FFT计划
        CUFFT_INVALID_PLAN：      句柄无效
        CUFFT_ALLOC_FAILED：      为计划分配GPU资源失败
        CUFFT_INVALID_VALUE：     向API传递了一个或多个无效参数。
        CUFFT_INTERNAL_ERROR：    检测到内部驱动程序错误
        CUFFT_SETUP_FAILED：      cuFFT库初始化失败
        CUFFT_INVALID_SIZE：      nx或批处理参数的大小不受支持
    */
    cufftHandle fft_2D_plan;
    int nx = N;
    int ny = M;
    cufftType type = CUFFT_C2C;

    CHECK_CUFFT_STATUS(cufftPlan2d(&fft_2D_plan, nx, ny, type));

    CHECK_CUFFT_STATUS(cufftExecC2C(fft_2D_plan, signal_c, spectrum_c, CUFFT_FORWARD));
    CHECK_CUFFT_STATUS(cufftDestroy(fft_2D_plan));
}

}   // namespace cufft_api



// signal, spectrum are device pointers
void fft_2D_cuda(const float* signal, float* spectrum, int M, int N) {
    // reBourne87::fft_2D_reBourne87_cuda(signal, spectrum, M, N);
    cufft_api::fft_2D_cufft_cuda(signal, spectrum, M, N);
}