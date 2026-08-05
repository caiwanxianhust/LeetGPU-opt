#include "utils.cuh"

#include <torch/extension.h>


/*
Write a GPU program that computes the histogram of an array of 32-bit integers. The histogram should count the number of occurrences of each integer value in the range [0, num_bins). You are given an input array input of length N and the number of bins num_bins.

The result should be an array of integers of length num_bins, where each element represents the count of occurrences of its corresponding index in the input array.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the histogram array.
Examples
Input: input = [0, 1, 2, 1, 0],  N = 5, num_bins = 3
Output: [2, 2, 1]
  
Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
Output: [0, 0, 0, 4, 0]
Constraints
1 ≤ N ≤ 100,000,000
0 ≤ input[i] < num_bins
1 ≤ num_bins ≤ 1024
*/


__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    __shared__ int shared_histogram[1024];

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int val = input[tid];
    if (tid < N) {
        atomicAdd(&shared_histogram[val], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&histogram[i], shared_histogram[i]);
    }
}


void histogram_cuda(const int* input, int* histogram, int N, int num_bins) {
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    histogram_kernel<<<grid, block>>>(input, histogram, N, num_bins);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor histogram(
    const torch::Tensor& input,
    int num_bins
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor histogram = torch::zeros({num_bins}, options);
    histogram_cuda(
        input.data_ptr<int>(),
        histogram.data_ptr<int>(),
        N,
        num_bins
    );
    return histogram;
}