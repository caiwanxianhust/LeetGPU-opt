#include "utils.cuh"


#include <cuda.h>

/*
Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix X of shape [M, N] containing quantized values and a scale matrix S of shape [ceil(M/T), ceil(N/T)], where T is the tile size.

For each element 
, the corresponding scale factor is 
 where 
 and 
. The output 
 should be computed as:

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the output buffer Y
Example 1:
Input:
M = 4, N = 4, TILE_SIZE = 2
X = [
  [10, 10,  5,  5],
  [10, 10,  5,  5],
  [ 2,  2,  8,  8],
  [ 2,  2,  8,  8]
]
S = [
  [0.5, 2.0],
  [4.0, 0.25]
]

Output:
Y = [
  [ 5.0,  5.0, 10.0, 10.0],
  [ 5.0,  5.0, 10.0, 10.0],
  [ 8.0,  8.0,  2.0,  2.0],
  [ 8.0,  8.0,  2.0,  2.0]
]
Explanation:
Tile (0,0) of X is multiplied by S[0,0] (0.5).
Tile (0,1) of X is multiplied by S[0,1] (2.0).
Tile (1,0) is multiplied by S[1,0] (4.0).
Tile (1,1) is multiplied by S[1,1] (0.25).
Constraints
1 ≤ M, N ≤ 8192
TILE_SIZE ∈ {16, 32, 64, 128}
*/


static __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void dequantize_kernel(const int M, const int N, const int TILE_SIZE, const float* X, const float* S, float* Y) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int tile_row = row / TILE_SIZE;
        int tile_col = col / TILE_SIZE;
        float scale = S[tile_row * div_up(N, TILE_SIZE) + tile_col];
        Y[row * N + col] = X[row * N + col] * scale;
    }
}





// X, S, Y are device pointers
void weight_dequantization_cuda(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    dim3 block(16, 16);
    dim3 grid(div_up(N, 16), div_up(M, 16));
    dequantize_kernel<<<grid, block>>>(M, N, TILE_SIZE, X, S, Y);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}