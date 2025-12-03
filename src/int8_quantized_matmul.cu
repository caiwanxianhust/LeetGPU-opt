#include "utils.cuh"

#include <cuda.h>
#include <cstdint>




/*
Implement a quantized matrix multiplication program for 8-bit signed integer matrices. Given two input matrices A of dimensions 
 and B of dimensions 
, quantization scales scale_A, scale_B, output scale scale_C, zero-points zero_point_A, zero_point_B, zero_point_C, compute:
 
where s_A = scale_A, z_A = zero_point_A, etc.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the output matrix C as int8
After accumulation in int32 and scaling in float32, values must be rounded to the nearest integer, shifted by zero_point_C, and clamped to the [-128, 127] range
Example 1:
     Input:
     A = [[1, 2],
          [3, 4]]
     B = [[5, 6],
          [7, 8]]
     M = 2, N = 2, K = 2
     scale_A = 0.1, scale_B = 0.2, scale_C = 0.05
     zero_point_A = 0, zero_point_B = 0, zero_point_C = 0
     
     Output:
     C = [[19, 22],
          [43, 50]]
     
Example 2:
     Input:
     A = [[1, 2]]
     B = [[3],
          [4]]
     M = 1, N = 1, K = 2
     scale_A = 1.0, scale_B = 1.0, scale_C = 1.0
     zero_point_A = 1, zero_point_B = 3, zero_point_C = 5
     
     Output:
     C = [[6]]
     
Constraints
1 ≤ M, N, K ≤ 4096
scale_A, scale_B, scale_C are positive floats
-128 ≤ zero_point_A, zero_point_B, zero_point_C ≤ 127
*/

static __device__ __host__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

template<int T>
__global__ void int8_quantized_matmul_kernel(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    //int T = 32;
    int tid = threadIdx.x;
    int tc = tid % T;
    int tr = tid / T;
    int col = blockIdx.x * T + tc;
    int row = blockIdx.y * T + tr;

    // load tile to smem:
    __shared__ int A_s[T][T];
    __shared__ int B_s[T][T];

    int numTiles = (K + T - 1) / T;

    int acc = 0;
    for (int t = 0; t < numTiles; t++) {
        int Acol = tc + t*T;
        int Brow = tr + t*T;
        A_s[tr][tc] = (Acol < K && row < M) ? static_cast<int>(A[row * K + Acol]) - zero_point_A : 0;
        B_s[tr][tc] = (col < N && Brow < K) ? static_cast<int>(B[Brow * N + col]) - zero_point_B : 0;
        __syncthreads();

        // Find dot product 
        if (col < N && row < M) {
            
            for (int i = 0; i < T; i++) {
                acc += A_s[tr][i] * B_s[i][tc];
                //acc += (static_cast<int>(A[row * K + i]) - zero_point_A) * (static_cast<int>(B[i * N + col]) - zero_point_B);
            }
        }
        __syncthreads();
    }
    if (col < N && row < M) {

        float scaled = (__int2float_rn(acc) * scale_A * scale_B / scale_C);
        acc = min(max(__float2int_rn(scaled) + zero_point_C, -128), 127);

        C[row*N + col] = acc;
    }
}

// A, B, C are device pointers
void int8_quantized_matmul_cuda(
    const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C) {
    constexpr int T = 32;

    dim3 block(T * T);
    dim3 grid(div_up(N, T), div_up(M, T));
    int8_quantized_matmul_kernel<T><<<grid, block>>>(
        A, B, C, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
} 