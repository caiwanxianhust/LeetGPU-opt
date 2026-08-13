#include "utils.cuh"
#include "cuda_kernels.cuh"


#include <cuda.h>
#include <torch/extension.h>

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


torch::Tensor weight_dequantization(
    const torch::Tensor& X,
    const torch::Tensor& S, 
    const int tile_size
) {
    TORCH_CHECK(X.ndimension() == 2, "X must be 2D");
    TORCH_CHECK(S.ndimension() == 2, "S must be 2D");
    TORCH_CHECK(X.device().is_cuda(), "X must be on CUDA");
    TORCH_CHECK(S.device().is_cuda(), "S must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be of type float32");
    TORCH_CHECK(S.dtype() == torch::kFloat32, "S must be of type float32");
    int M = X.size(0);
    int N = X.size(1);
    TORCH_CHECK(((M + tile_size - 1) / tile_size) == S.size(0), "Invalid shape for S");
    TORCH_CHECK(((N + tile_size - 1) / tile_size) == S.size(1), "Invalid shape for S");

    torch::Tensor output = torch::empty({M, N}, X.options());

    weight_dequantization_cuda(
        X.data_ptr<float>(),
        S.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, tile_size
    );
    return output;
}