#include "utils.cuh"


/*
Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as:

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in vector output
Use 
 as the leaky coefficient
Example 1:
  Input:  x = [1.0, -2.0, 3.0, -4.0]
  Output: y = [1.0, -0.02, 3.0, -0.04]
Example 2:
  Input:  x = [-1.5, 0.0, 2.5, -3.0]
  Output: y = [-0.015, 0.0, 2.5, -0.03]
Constraints
1 ≤ N ≤ 100,000,000
-1000.0 ≤ input[i] ≤ 1000.0
*/

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        output[tid] = (input[tid] > 0) ? input[tid] : 0.01f * input[tid];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void leaky_relu_cuda(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
