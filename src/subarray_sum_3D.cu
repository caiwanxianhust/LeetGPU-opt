#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>


/*
Implement a program that computes the sum of a 3D subarray of 32-bit integers. You are given an input 3D array input of length N x M x K, and two depth indices S_DEP and E_DEP and two row indices S_ROW and E_ROW and two column indices S_COL and E_COL. S_DEP, E_DEP, S_ROW, E_ROW, S_COL and E_COL are inclusive, 0-based start and end indices — compute the sum of input[S_DEP..E_DEP][S_ROW..E_ROW][S_COL..E_COL].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input:  input = [[[1, 2, 3],
                  [4, 5, 1]],
                 [[1, 1, 1],
                  [2, 2, 2]]]
        N = 2, M = 2, K = 3
        S_DEP = 0, E_DEP = 1, S_ROW = 0, E_ROW = 0, S_COL = 1, E_COL = 2
Output: output = 7
Example 2:
Input:  input = [[[5, 10],
                  [5, 2],
                  [2, 2]]]
        N = 1, M = 3, K = 2 
        S_DEP = 0, E_DEP = 0, S_ROW = 0, E_ROW = 2, S_COL = 1, E_COL = 1
Output: output = 14
Constraints
1 ≤ N, M, K ≤ 500
1 ≤ input[i] ≤ 10
0 ≤ S_DEP ≤ E_DEP ≤ N - 1
0 ≤ S_ROW ≤ E_ROW ≤ M - 1
0 ≤ S_COL ≤ E_COL ≤ K - 1
*/


static __global__ void zeros_init_kernel(int* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        output[i] = 0;
    }
}

__global__ void subarray_sum_3D_kernel(
    const int* input, int* output, int N, int M, int K,
    int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {
    int dep_idx = blockIdx.y + S_DEP;
    int row_idx = blockIdx.x + S_ROW;
    int col_idx = threadIdx.x + S_COL;
    if (dep_idx > E_DEP || row_idx > E_ROW || col_idx > E_COL) { 
        return;
    }
    int vsum = 0;
    for (int i = col_idx; i <= E_COL; i+= blockDim.x) {
        vsum += input[dep_idx * M * K + row_idx * K + i];
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        atomicAdd(output, vsum);
    }
}



void subarray_sum_3D_cuda(
    const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL) {

    dim3 block(256);
    dim3 grid(E_ROW - S_ROW + 1, E_DEP - S_DEP + 1);
    zeros_init_kernel<<<1, 256>>>(output, 1);
    subarray_sum_3D_kernel<<<grid, block>>>(input, output, N, M, K, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
