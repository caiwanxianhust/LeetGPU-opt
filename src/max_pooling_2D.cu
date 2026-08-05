#include "utils.cuh"

#include <cuda.h>
#include <cfloat>
#include <torch/extension.h>

/*
Implement a 2D max pooling operation for image/feature map downsampling. The program should take an input tensor and produce an output tensor by applying max pooling with specified kernel size, stride, and padding.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in tensor output
Max Pooling Operation
For each output position (n, c, h_out, w_out), compute the maximum value over the corresponding input window:
output[n, c, h_out, w_out] = max(input[n, c, h:h+kernel_size, w:w+kernel_size])
where h = h_out * stride and w = w_out * stride

Example 1:
Input:  input = [[[[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0],
                   [7.0, 8.0, 9.0]]]]
        kernel_size = 2
        stride = 1
        padding = 0
Output: output = [[[[5.0, 6.0],
                    [8.0, 9.0]]]]
Example 2:
Input:  input = [[[[1.0, 2.0, 3.0, 4.0, 5.0],
                   [6.0, 7.0, 8.0, 9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0],
                   [21.0, 22.0, 23.0, 24.0, 25.0]]]]
        kernel_size = 3
        stride = 1
        padding = 1
Output: output = [[[[7.0, 8.0, 9.0, 10.0, 10.0],
                    [12.0, 13.0, 14.0, 15.0, 15.0],
                    [17.0, 18.0, 19.0, 20.0, 20.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0],
                    [22.0, 23.0, 24.0, 25.0, 25.0]]]]
Constraints
1 ≤ N ≤ 100 (batch size)
1 ≤ C ≤ 512 (channels)
1 ≤ H, W ≤ 1024 (height, width)
1 ≤ kernel_size ≤ 16
1 ≤ stride ≤ 16
0 ≤ padding ≤ 16
Input and output tensors use float32 precision
*/


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void max_pooling_2D_kernel(const float* input, float* output,
                                    int N, int C, int H, int W,
                                    int H_out, int W_out,
                                    int kernel_size, int stride, int padding) { 
    int batch_id = blockIdx.z / C;
    int channel_id = blockIdx.z % C;
    float *out_ptr = output + batch_id * C * H_out * W_out + channel_id * H_out * W_out;
    const float *in_ptr = input + batch_id * C * H * W + channel_id * H * W;

    int row_out = blockIdx.y * blockDim.y + threadIdx.y;
    int col_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_out < H_out && col_out < W_out) { 
        int row_in_start = row_out * stride - padding;
        int col_in_start = col_out * stride - padding;
        int row_in_end = row_in_start + kernel_size;
        int col_in_end = col_in_start + kernel_size;
        float max_val = -FLT_MAX;
        for (int row_in = row_in_start; row_in < row_in_end; row_in++) {
            for (int col_in = col_in_start; col_in < col_in_end; col_in++) {
                if (row_in >= 0 && row_in < H && col_in >= 0 && col_in < W) {
                    float val = in_ptr[row_in * W + col_in];
                    max_val = max(max_val, val);
                }
            }
        }
        out_ptr[row_out * W_out + col_out] = max_val;
    }
}




// input, output are device pointers (i.e. pointers to memory on the GPU)
void max_pooling_2D_cuda(const float* input, float* output,
                      int N, int C, int H, int W,
                      int kernel_size, int stride, int padding) {
    // the shape of output
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    dim3 block(32, 32);
    dim3 grid(div_up(W_out, block.x), div_up(H_out, block.y), N * C);
    max_pooling_2D_kernel<<<grid, block>>>(input, output, N, C, H, W, H_out, W_out, kernel_size, stride, padding);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor max_pooling_2D(
    const torch::Tensor& input,
    int kernel_size,
    int stride,
    int padding
) { 
    TORCH_CHECK(input.ndimension() == 4, "Input must be 4D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32")

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    torch::Tensor output = torch::empty({N, C, H_out, W_out}, input.options());

    max_pooling_2D_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, stride, padding
    );
    return output;
}