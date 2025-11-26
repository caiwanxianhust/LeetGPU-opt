#include "utils.cuh"


/*
1D Convolution
Easy
Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of two arrays:

input: A 1D array of 32-bit floating-point numbers.
kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.
The output should be written to the output array, which will have a size of input_size - kernel_size + 1.
The convolution operation is defined mathematically as:


Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output
Example 1:
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]

Example 2:
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]

Constraints
1 ≤ input_size ≤ 1,500,000
1 ≤ kernel_size ≤ 2047
kernel_size ≤ input_size
*/


__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    __shared__ float s_kernel[2048];
    __shared__ float s_input[256 + 2048 - 1];
    for (int i=threadIdx.x; i<kernel_size; i+=blockDim.x) {
        s_kernel[i] = kernel[i];
    }
    for (int i=threadIdx.x; i<(256+2048-1); i+=blockDim.x) {
        s_input[i] = (blockIdx.x * blockDim.x + i) < input_size ? input[blockIdx.x * blockDim.x + i] : 0.0f;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    float vsum = 0.0f;
    if (tid < output_size) {
        for (int j=0; j<kernel_size; j++) {
            vsum += s_input[threadIdx.x + j] * s_kernel[j]; 
        }
        output[tid] = vsum;
    }

}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void convolution_1d_cuda(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}