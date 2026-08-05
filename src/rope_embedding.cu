#include "utils.cuh"

#include <cuda.h>
#include <torch/extension.h>


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


torch::Tensor rope_embedding(
    const torch::Tensor& input,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(cos_table.ndimension() == 2, "cos_table must be 2D");
    TORCH_CHECK(sin_table.ndimension() == 2, "sin_table must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(cos_table.device().is_cuda(), "cos_table must be on CUDA");
    TORCH_CHECK(sin_table.device().is_cuda(), "sin_table must be on CUDA");
    TORCH_CHECK(input.size(0) == cos_table.size(0), "Input and cos_table must have the same number of rows");
    TORCH_CHECK(input.size(0) == sin_table.size(0), "Input and sin_table must have the same number of rows");
    TORCH_CHECK(input.size(1) == cos_table.size(1), "Input and cos_table must have the same number of columns");
    TORCH_CHECK(input.size(1) == sin_table.size(1), "Input and sin_table must have the same number of columns");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(cos_table.dtype() == torch::kFloat32, "cos_table must be of type float32");
    TORCH_CHECK(sin_table.dtype() == torch::kFloat32, "sin_table must be of type float32");
    TORCH_CHECK(input.size(1) % 2 == 0, "input must have an even number of columns");

    int M = input.size(0);
    int D = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({M, D}, options);

    rope_embedding_cuda(
        input.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        output.data_ptr<float>(),
        M, D
    );
    
    return output;
}