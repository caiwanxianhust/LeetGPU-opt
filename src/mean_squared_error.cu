#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>


/*
Implement a GPU program to calculate the Mean Squared Error (MSE) between predicted values and target values. Given two arrays of equal length, predictions and targets, compute:
 
 
 
 
 
 
 
 
where N is the number of elements in each array.

Implementation Requirements
External libraries are not permitted.
The solve function signature must remain unchanged.
The final result must be stored in the mse variable.
Example 1:
  Input:  predictions = [1.0, 2.0, 3.0, 4.0]
          targets = [1.5, 2.5, 3.5, 4.5]
  Output: mse = 0.25
Example 2:
  Input:  predictions = [10.0, 20.0, 30.0]
          targets = [12.0, 18.0, 33.0]
  Output: mse = 5.67
Constraints
1 ≤ N ≤ 100,000,000
-1000.0 ≤ predictions[i], targets[i] ≤ 1000.0
*/


__global__ void mean_squared_error_kernel(
    const float* predictions, const float* targets, float* diff_squared, int N
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float vsum = 0.0f;
    float temp;
    for (int i=tid; i<N; i+=blockDim.x*gridDim.x) {
        temp = predictions[i] - targets[i];
        vsum += temp * temp;
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);
    if (threadIdx.x == 0) {
        diff_squared[blockIdx.x] = vsum / (float)N;
    }
}


__global__ void mean_reduce_kernel(float* diff_squared, float* mse, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float temp = 0.0f;
    for (int i=tid; i<N; i+=blockDim.x*gridDim.x) {
        temp += diff_squared[i];
    }
    __syncthreads();
    temp = block_all_reduce_sum(temp);
    if (threadIdx.x == 0) {
        *mse = temp;
    }
}

void mean_squared_error_cuda(
    const float* predictions, const float* targets, float* mse, int N
) {
    int block_size = 256;
    int grid_size;
    if (N <= block_size) {
        grid_size = 1;
    }
    else if (block_size * 1024 < N) {
        grid_size = (N + block_size - 1) / block_size;
    }
    else {
        grid_size = 1024;
    }
    float *diff_squared;
    CHECK_CUDA_ERROR(cudaMalloc(&diff_squared, sizeof(float) * grid_size));
    mean_squared_error_kernel<<<grid_size, block_size>>>(predictions, targets, diff_squared, N);
    mean_reduce_kernel<<<1, 1024>>>(diff_squared, mse, grid_size);
    CHECK_CUDA_ERROR(cudaFree(diff_squared));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
