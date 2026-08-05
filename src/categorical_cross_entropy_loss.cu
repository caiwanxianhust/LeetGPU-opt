#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>


/**
Implement a GPU program to calculate the categorical cross-entropy loss for a batch of predictions. Given a matrix of predicted logits 
 of size 
 and a vector of true class labels true_labels of size 
, compute the average cross-entropy loss over the batch. The loss for a single sample 
 with logits 
 and true label 
 is calculated using the numerically stable formula:

 The final output stored in the loss variable should be the average loss over the 
 samples:
 
The input parameters are logits, true_labels, N (number of samples), and C (number of classes). The result should be stored in loss (a pointer to a single float).

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result (average loss) must be stored in loss
Example 1:
Input:  N = 2, C = 3
        logits = [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]]
        true_labels = [1, 1]
Output: loss = [0.3548926]
Example 2:
Input:  N = 3, C = 4
        logits = [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]
        true_labels = [3, 0, 1]
Output: loss = [0.98820376]
Constraints
1 ≤ N ≤ 10,000
2 ≤ C ≤ 1,000
-10.0 ≤ logits[i, j] ≤ 10.0
0 ≤ true_labels[i] ≤ C
 */


__global__ void categorical_cross_entropy_loss_kernel(
    const float* logits, const int* true_labels, float* tmp_loss, int N, int C
) {
    const int offset_logits = blockIdx.x * C;
    float z_jk = 0.0f;
    float z_jy = logits[offset_logits + true_labels[blockIdx.x]];
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        z_jk += expf(logits[offset_logits + i]);
    }
    __syncthreads();

    z_jk = block_all_reduce_sum(z_jk);
    if (threadIdx.x == 0) {
        tmp_loss[blockIdx.x] = logf(z_jk) - z_jy;
    }
}


__global__ void avg_reduce_loss_kernel(float* tmp_loss, float* loss, int N) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        sum += tmp_loss[i];
    }
    __syncthreads();

    sum = block_all_reduce_sum(sum);
    if (threadIdx.x == 0) {
        *loss = sum / N;
    }
}


void avg_categorical_cross_entropy_loss_cuda(
    const float* logits, const int* true_labels, float* loss, int N, int C
) {
    // Launch the kernel with appropriate grid and block dimensions
    float *tmp_loss;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&tmp_loss, N * sizeof(float)));
    dim3 block(256);
    dim3 grid(N);
    categorical_cross_entropy_loss_kernel<<<grid, block>>>(logits, true_labels, tmp_loss, N, C);

    avg_reduce_loss_kernel<<<1, block>>>(tmp_loss, loss, N);

    // Check for errors in kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    // Free temporary memory
    CHECK_CUDA_ERROR(cudaFree(tmp_loss));
}


torch::Tensor avg_categorical_cross_entropy_loss(
    const torch::Tensor& logits,
    const torch::Tensor& true_labels,
    int N, int C
) {
    TORCH_CHECK(logits.ndimension() == 1, "Input logits must be 1D");
    TORCH_CHECK(true_labels.ndimension() == 1, "Input true_labels must be 1D");
    TORCH_CHECK(logits.device().is_cuda(), "Input logits must be on CUDA");
    TORCH_CHECK(true_labels.device().is_cuda(), "Input true_labels must be on CUDA");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input logits must be of type float32");
    TORCH_CHECK(true_labels.dtype() == torch::kInt32, "Input true_labels must be of type int32");
    TORCH_CHECK(logits.size(0) == N * C, "Input logits must be a 1D matrix")
    TORCH_CHECK(true_labels.size(0) == N, "Input true_labels must be a 1D matrix")
    torch::Tensor loss = torch::empty({1}, logits.options());
    avg_categorical_cross_entropy_loss_cuda(
        logits.data_ptr<float>(),
        true_labels.data_ptr<int>(),
        loss.data_ptr<float>(),
        N, C
    );
    return loss;
}