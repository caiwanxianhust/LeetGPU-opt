#include "utils.cuh"

#include <cfloat>
#include <cub/cub.cuh>
#include <torch/extension.h>

/*
Write a program that sorts an array of 32-bit floating-point numbers in ascending order. You are free to choose any sorting algorithm.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The sorted result must be stored back in the input data array
Example
Input: data = [5.0, 2.0, 8.0, 1.0, 9.0, 4.0], N = 6
Output: data = [1.0, 2.0, 4.0, 5.0, 8.0, 9.0]
Constraints
1 ≤ N ≤ 1,000,000
*/

namespace bitonic_sort {
__global__ void bitonic_sort_step(float* data, int N, int j, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int l = i ^ j;
    if (l > i && i < N && l < N) {
        if ((i & k) == 0) {
            if (data[i] > data[l]) {
                float tmp = data[i];
                data[i] = data[l];
                data[l] = tmp;
            }
        }
        else {
            if (data[i] < data[l]) {
                float tmp = data[i];
                data[i] = data[l];
                data[l] = tmp;
            }
        }
    }
}


__device__ __host__ int next_power_of_two(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}


// ===================== 主机端入口函数（data为设备端指针）=====================
void sort_bitonic_cuda(float* data, int N) {
    printf("bitonic_sort\n");
    if (N <= 1) return;

    int paddedN = next_power_of_two(N);
    float* sortData = NULL;

    if (paddedN > N) {
        cudaMalloc(&sortData, paddedN * sizeof(float));
        cudaMemcpy(sortData, data, N * sizeof(float), cudaMemcpyDeviceToDevice);

        float* hostPadding = (float*)malloc((paddedN - N) * sizeof(float));
        for (int i = 0; i < paddedN - N; i++) hostPadding[i] = FLT_MAX;

        cudaMemcpy(&sortData[N], hostPadding, (paddedN - N) * sizeof(float), cudaMemcpyHostToDevice);
        free(hostPadding);

    }
    else sortData = data;

    const int blockSize = 256;
    int gridSize = (paddedN + blockSize - 1) / blockSize;

    for (int k = 2; k <= paddedN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step << <gridSize, blockSize >> > (sortData, paddedN, j, k);
            cudaDeviceSynchronize();
        }
    }

    if (paddedN > N) {
        cudaMemcpy(data, sortData, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(sortData);
    }
}

} // namespace bitonic_sort


namespace cub_sort {

void sort_cub_cuda(float* data, int N) { 
    printf("cub_sort\n");
    float *sortData;
    CHECK_CUDA_ERROR(cudaMalloc(&sortData, N * sizeof(float)));

    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, data, sortData, N);

    // Allocate temporary storage
    CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(
    d_temp_storage, temp_storage_bytes, data, sortData, N);

    // Copy the sorted data back to the original array
    CHECK_CUDA_ERROR(cudaMemcpy(data, sortData, N * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaFree(sortData));
    CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
}

} // namespace cub_sort


void sort_cuda(float* data, int N) {
    bitonic_sort::sort_bitonic_cuda(data, N);
    // cub_sort::sort_cub_cuda(data, N);
}


void sorting(const torch::Tensor& input) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    sort_cuda(input.data_ptr<float>(), input.size(0));
    return;
}