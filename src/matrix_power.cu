#include "utils.cuh"

#include <cuda.h>
#include <stdio.h>


/*
Implement a GPU program that raises a square matrix 
 of size 
 to an integer power 
.
The solve function receives a flattened input matrix input (row-major order), an empty output matrix output of the same size, the dimension N, and the exponent P.
You must compute 
 where matrix multiplication is standard dense multiplication over 32-bit floating point numbers.

Implementation Requirements
External libraries are not permitted.
The solve function signature must remain unchanged.
The final result must be written to the output array in row-major order.
Example 1:
  Input:
    input  = [[1.0, 2.0],
              [3.0, 4.0]]
    N      = 2
    P      = 3
  Output:
    output = [[37.0, 54.0],
              [81.0, 118.0]]
  
Example 2:
  Input:
    input  = [[1.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 0.0]]
    N      = 3
    P      = 2
  Output:
    output = [[7.0, 0.0, 2.0],
              [0.0, 1.0, 0.0],
              [3.0, 0.0, 6.0]]
  
*/


namespace { 
    constexpr int tile_size = 32;
}


static __host__ __device__ __forceinline__ int div_up(int m, int n) {
    return (m + n - 1) / n;
}


__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) { 
    __shared__ float A_tile[tile_size * tile_size];
    __shared__ float B_tile[tile_size * tile_size];

    int tx = threadIdx.x % tile_size;
    int ty = threadIdx.x / tile_size;
    int row_c = blockIdx.y * tile_size + ty;
    int col_c = blockIdx.x * tile_size + tx;

    float acc = 0.0f;

    for (int k = 0; k < N; k += tile_size) {
        for (int i = threadIdx.x; i < tile_size * tile_size; i += blockDim.x) {
            int row_a = (i / tile_size) + (blockIdx.y * tile_size);
            int col_a = (i % tile_size) + k;
            if (row_a < N && col_a < N) {
                A_tile[i] = A[row_a * N + col_a];
            } else {
                A_tile[i] = 0.0f;
            }
        }

        for (int i = threadIdx.x; i < tile_size * tile_size; i += blockDim.x) {
            int row_b = k + (i / tile_size);
            int col_b = (i % tile_size) + blockIdx.x * tile_size;
            if (row_b < N && col_b < N) {
                B_tile[i] = B[row_b * N + col_b];
            } else {
                B_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        if (row_c < N && col_c < N) {
            for (int i = 0; i < tile_size; ++i) {
                acc += A_tile[ty * tile_size + i] * B_tile[i * tile_size + tx];
            }
        }
        __syncthreads();
    }

    if (row_c < N && col_c < N) {
        C[row_c * N + col_c] = acc;
    }
}




// input, output are device pointers
void matrix_power_cuda(const float* input, float* output, int N, int P) {
    printf("matrix_power_cuda(input, output, %d, %d)\n", N, P);
    if (P == 1) {
        CHECK_CUDA_ERROR(cudaMemcpy(output, input, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }
    dim3 block(tile_size * tile_size);
    dim3 grid(div_up(N, tile_size), div_up(N, tile_size));
    if (P == 2) {
        // 直接调用矩阵乘法核函数
        matrix_multiply_kernel<<<grid, block>>>(input, input, output, N);
        return;
    }
    int half_p = P / 2;
    // 是否偶数
    if (P % 2 == 0)
    {
        float *d_buf;
        size_t buf_size = N * N * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));

        matrix_power_cuda(input, d_buf, N, half_p);
        // 计算 output = output * output
        // 直接调用矩阵乘法核函数
        matrix_multiply_kernel<<<grid, block>>>(d_buf, d_buf, output, N);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaFree(d_buf));
        return;
    }
    else {
        float *d_buf;
        size_t buf_size = 2 * N * N * sizeof(float);
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
        float *d_out = d_buf + N * N;

        matrix_power_cuda(input, d_buf, N, half_p);
        matrix_multiply_kernel<<<grid, block>>>(d_buf, d_buf, d_out, N);
        CHECK_CUDA_ERROR(cudaGetLastError());
        matrix_multiply_kernel<<<grid, block>>>(input, d_out, output, N);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaFree(d_buf));
        return;
    }
} 