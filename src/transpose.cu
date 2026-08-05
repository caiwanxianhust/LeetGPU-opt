#include "utils.cuh"

#include <torch/extension.h>

/*
Write a program that transposes a matrix of 32-bit floating point numbers on a GPU. The transpose of a matrix switches its rows and columns. Given a matrix 
 of dimensions 
, the transpose 
 will have dimensions 
. All matrices are stored in row-major format.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the matrix output

Constraints
1 ≤ rows, cols ≤ 8192
Input matrix dimensions: rows × cols
Output matrix dimensions: cols × rows
*/


__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    __shared__ float s_buf[32][33];
    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;
    if (x < cols && y < rows) {
        s_buf[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    x = bx + threadIdx.y;
    y = by + threadIdx.x;
    if (x < cols && y < rows) {
        output[x * rows + y] = s_buf[threadIdx.x][threadIdx.y];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void matrix_transpose_cuda(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor matrix_transpose(
    const torch::Tensor& input
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int rows = input.size(0), cols = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({cols, rows}, options);
    matrix_transpose_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );
    return output;
}
