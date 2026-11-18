#include "utils.cuh"

#include <torch/extension.h>



static __device__ float warp_inclusive_scan(float val) {
    int lane_id = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset *= 2) {
        float tmp = __shfl_up_sync(0xffffffff, val, offset);
        if (lane_id >= offset) val += tmp;
    }
    return val;
}

static __device__ float block_inclusive_scan(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_inclusive_scan(val);
    if (lane == 31) shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        float warp_sum = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        warp_sum = warp_inclusive_scan(warp_sum);
        if (lane < (blockDim.x >> 5)) shared[lane] = warp_sum;
    }
    __syncthreads();

    if (warp_id > 0) val += shared[warp_id - 1];
    return val;
}



// ---- K1: 每行一块，沿 W 做含进位的前缀和 ----
__global__ void sat_row_scan(const float* in, float* out, int H, int W) {
    int row = blockIdx.x;  
    if (row >= H) return;
    const float* in_row  = in  + row * W;
    float*       out_row = out + row * W;   
    __shared__ float sh_carry;
    if (threadIdx.x == 0) sh_carry = 0.0;  
    __syncthreads();

    for (int t = 0; t < W; t += blockDim.x) {
        int idx = t + threadIdx.x;
        float val = (idx < W) ? in_row[idx] : 0.0f;
        __syncthreads();

        val = block_inclusive_scan(val);
        val += sh_carry;
        if (idx < W) {
            out_row[idx] = val;
        }
        __syncthreads(); 
        if (threadIdx.x == blockDim.x - 1) {
            sh_carry = val;
        }
    }
}

// ---- K2: 每 BW 列一块，沿 H 做含进位向量的前缀和 ----
__global__ void sat_col_scan(float* out, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < W) {
        float val = out[0 * W + col];
        for (int row = 1; row < H; ++row) {
            float prev = out[row * W + col];
            val += prev;
            out[row * W + col] = val;
        }
    }
}


void integral_image_cuda(const float* input, float* output, int H, int W) 
{
    sat_row_scan<<<H, 256>>>(input, output, H, W);
    sat_col_scan<<<(W + 255) / 256, 256>>>(output, H, W);
}


torch::Tensor integral_image(const torch::Tensor& input) {
    TORCH_CHECK(input.dim() == 2, "Input tensor must be 2D");
    int H = input.size(0);
    int W = input.size(1);
    auto output = torch::zeros({H, W}, input.options());
    integral_image_cuda(input.data_ptr<float>(), output.data_ptr<float>(), H, W);
    return output;
}
