#include "utils.cuh"
#include <cuda.h>

/*
Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil: each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom, left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged from the input to the output.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in output
Read exclusively from input and write exclusively to output (do not update input)

Constraints
1 ≤ rows, cols ≤ 16,384
Input values are in the range [-100, 100]
All values are 32-bit floats
Performance is measured with rows = 8,192, cols = 8,192
*/

static __device__ __host__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static constexpr int tile_size = 32;

__global__ void jacobi_stencil_2D_kernel(const float* input, float* output, int rows, int cols) {
    const int row_id = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_id = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_inp[tile_size][tile_size];
    s_inp[threadIdx.y][threadIdx.x] = (row_id < rows && col_id < cols) ? input[row_id * cols + col_id] : 0.0f;
    __syncthreads();

    // boundary cells are copied unchanged
    if (row_id >= rows || col_id >= cols) return;  // 提前退出超出边界的线程

    // boundary cells are copied unchanged
    if (row_id == 0 || col_id == 0 || row_id == rows - 1 || col_id == cols - 1) {
        output[row_id * cols + col_id] = input[row_id * cols + col_id];
    }
    else if (row_id > 0 && col_id > 0 && row_id < rows - 1 && col_id < cols - 1) { 
        float left = (threadIdx.x == 0) ? input[row_id * cols + col_id - 1] : s_inp[threadIdx.y][threadIdx.x - 1];
        float right = (threadIdx.x == tile_size - 1) ? input[row_id * cols + col_id + 1] : s_inp[threadIdx.y][threadIdx.x + 1];
        float top = (threadIdx.y == 0) ? input[(row_id - 1) * cols + col_id] : s_inp[threadIdx.y - 1][threadIdx.x];
        float bottom = (threadIdx.y == tile_size - 1) ? input[(row_id + 1) * cols + col_id] : s_inp[threadIdx.y + 1][threadIdx.x];
        float val = 0.25f * (left + right + top + bottom);
        output[row_id * cols + col_id] = val;
    }
}


// input, output are device pointers
void jacobi_stencil_2D_cuda(const float* input, float* output, int rows, int cols) 
{
    dim3 block(tile_size, tile_size);
    dim3 grid(div_up(cols, tile_size), div_up(rows, tile_size));
    jacobi_stencil_2D_kernel<<<grid, block>>>(input, output, rows, cols);
}