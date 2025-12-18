#include "utils.cuh"

#include "cuda_kernels.cuh"

/*
Write a GPU program that counts the number of elements with the integer value k in an 2D array of 32-bit integers. The program should count the number of elements with k in an 2D array. You are given an input 2D array input of length N x M and integer k.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: input [[1, 2, 3],
              [4, 5, 1]]
       k = 1
Output: output = 2
Example 2:
Input: input [[5, 10],
              [5, 2]]
       k = 1
Output: output = 0
Constraints
1 ≤ N, M ≤ 10,000
1 ≤ input[i], k ≤ 100
*/

// __device__ int warp_all_reduce_sum(int val) {

//     for (int mask = 16; mask > 0; mask >>= 1) {
//         val += __shfl_xor_sync(0xffffffff, val, mask);
//     }
//     return val;
// }


// __device__ int block_all_reduce_sum(int val) {
//     __shared__ int result;
//     __shared__ int s_buf[32];
//     int warp_id = threadIdx.x >> 5;
//     int lane_id = threadIdx.x & 0x1f;

//     val = warp_all_reduce_sum(val);
//     if (lane_id == 0) {
//         s_buf[warp_id] = val;
//     }
//     __syncthreads();

//     val = threadIdx.x < (blockDim.x >> 5) ? s_buf[threadIdx.x] : 0;
//     val = warp_all_reduce_sum(val);

//     if (threadIdx.x == 0) {
//         result = val;
//     }
//     __syncthreads();
//     return result;
// }

__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {

    int row_id = blockIdx.x;
    int vsum = 0;

    for (int i=threadIdx.x; i<M; i+=blockDim.x) {
        vsum += (input[row_id * M + i] == K) ? 1 : 0;
    }

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void count_2d_equal_cuda(const int* input, int* output, int N, int M, int K) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(N);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}