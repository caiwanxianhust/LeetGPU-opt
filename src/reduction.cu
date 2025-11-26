#include "utils.cuh"

#include "cuda_kernels.cuh"

/**
Write a GPU program that performs parallel reduction on an array of 32-bit floating point numbers to compute their sum. The program should take an input array and produce a single output value containing the sum of all elements.

Implementation Requirements
Use only GPU native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Output: 36.0
Example 2:
Input: [-2.5, 1.5, -1.0, 2.0]
Output: 0.0
Constraints
1 ≤ N ≤ 100,000,000
-1000.0 ≤ input[i] ≤ 1000.0
The final sum will always fit within a 32-bit float
 */

// __device__ float warp_all_reduce_sum(float val) {
//     #pragma unroll
//     for (int mask=16; mask > 0; mask >>= 1) {
//         val += __shfl_xor_sync(0xffffffff, val, mask);
//     }
//     return val;
// }


// __device__ float block_all_reduce_sum(float val) {
//     int warp_id = threadIdx.x >> 5;
//     int lane_id = threadIdx.x & 0x1f;

//     __shared__ float result;
//     __shared__ float s_buf[32];

//     val = warp_all_reduce_sum(val);
//     if (lane_id == 0) {
//         s_buf[warp_id] = val;
//     }
//     __syncthreads();

//     val = threadIdx.x < (blockDim.x >> 5) ? s_buf[threadIdx.x] : 0.0f;
//     val = warp_all_reduce_sum(val);
//     if (threadIdx.x == 0) {
//         result = val;
//     }
//     __syncthreads();
//     return result;
// }


__global__ void reduce_sum_kernel(const float* input, float* output, int N) {

    float vsum = 0.0f;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i=tid; i<N; i+=blockDim.x * gridDim.x) {
        vsum += input[i];
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);
    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}
    

// input, output are device pointers
void reduce_sum_cuda(const float* input, float* output, int N) {  
    int block_size = 1024;
    int grid_size;
    if (N <= block_size) {
        grid_size = 1;
    }
    else if (N < 128 * block_size) {
        grid_size = (N + block_size - 1) / block_size;
    }
    else {
        grid_size = 128;
    }

    reduce_sum_kernel<<<grid_size, block_size>>>(input, output, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}