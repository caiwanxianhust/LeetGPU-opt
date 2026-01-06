#include "utils.cuh"

#include <cuda.h>


/*
Implement a GPU program that computes the Rotary Positional Embedding (RoPE) for a batch of query vectors. RoPE is a method for encoding positional information in transformer models by rotating the query and key vectors using precomputed cosine and sine components.

Mathematically, given a query vector 
 and corresponding cosine and sine vectors, the operation is defined as:

Where 
 denotes element-wise multiplication. The 
 operation swaps the first and second halves of the vector and negates the first half. For a vector of dimension 
:

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The input tensors Q, cos, and sin have shape (M, D), where M is the number of tokens and D is the head dimension
D (head dimension) is guaranteed to be an even number
The final result must be stored in the output variable with the same shape (M, D)
Example 1:
Input:  Q   = [[1.0, 2.0, 3.0, 4.0],
               [1.0, 1.0, 1.0, 1.0]]
        Cos = [[1.0, 1.0, 1.0, 1.0],
               [0.0, 0.0, 0.0, 0.0]]
        Sin = [[0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0]]
Output: result = [[1.0, 2.0, 3.0, 4.0],
                  [-1.0, -1.0, 1.0, 1.0]]
        (Row 0 is identity via Cos; Row 1 is rotated via Sin)
Constraints
Q, cos, and sin have identical dimensions
D % 2 == 0
1 ≤ M, D ≤ 10,000
*/

__global__ void rope_kernel(const float* Q, const float* cos, const float* sin, float* output, int M, int D) {
    float val, val_half;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        val = Q[blockIdx.x * D + i];
        val_half = (i < D / 2) ? -Q[blockIdx.x * D + i + D / 2] : Q[blockIdx.x * D + i - D / 2];
        output[blockIdx.x * D + i] = val * cos[blockIdx.x * D + i] + val_half * sin[blockIdx.x * D + i];
    }
}


// Q, cos, sin, output are device pointers
void rope_embedding_cuda(float* Q, float* cos, float* sin, float* output, int M, int D) {
    rope_kernel<<<M, 256>>>(Q, cos, sin, output, M, D);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
