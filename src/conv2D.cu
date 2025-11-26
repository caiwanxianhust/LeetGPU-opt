#include "utils.cuh"


/*
Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of:

input: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
kernel: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.
The output should be written to the output matrix (also a 1D array in row-major order). The output matrix will have dimensions:

output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output

Constraints
1 ≤ input_rows, input_cols ≤ 3072
1 ≤ kernel_rows, kernel_cols ≤ 31
kernel_rows ≤ input_rows
kernel_cols ≤ input_cols
*/


__device__ float matrix_inner_product(const float* input, const float* kernel,
                                 int rows, int cols, int stride) {
    float result = 0.0f;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result += input[i * stride + j] * kernel[i * cols + j];
        }
    }
    return result;
}


__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) { 
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float *input_ptr = input + row * input_cols + col;
    if (row < output_rows && col < output_cols) { 
        output[row * output_cols + col] = matrix_inner_product(input_ptr, kernel, kernel_rows, kernel_cols, input_cols);
    }
}


// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void convolution_2d_cuda(const float* input, const float* kernel, float* output, 
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    dim3 block(16, 16);
    dim3 grid((output_cols + block.x - 1) / block.x, (output_rows + block.y - 1) / block.y);
    convolution_2d_kernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}