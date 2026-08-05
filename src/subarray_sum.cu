#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Implement a program that computes the sum of a subarray of 32-bit integers. You are given an input array input of length N, and two indices S and E. S and E are inclusive, 0-based start and end indices — compute the sum of input[S..E].

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the output variable
Example 1:
Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
Output: output = 6
Example 2:
Input: input = [1, 2, 3, 4], S = 0, E = 3
Output: output = 10
Constraints
1 ≤ N ≤ 100,000,000
1 ≤ input[i] ≤ 10
0 ≤ S ≤ E ≤ N - 1
*/


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


// launch configuration: one block, 1024 threads
__global__ void short_vector_reduce_sum_kernel(const int* input, int* output, int num_elems) {
    int vsum = 0;

    for (int i=threadIdx.x; i<num_elems; i+=blockDim.x) {
        vsum += input[i];
    }
    __syncthreads();

    vsum = block_all_reduce_sum(vsum);

    if (threadIdx.x == 0) {
        output[0] = vsum;
    }
}

__global__ void vector_reduce_sum_kernel(const int* input, int* output, int num_elems, int nums_per_block) { 
    int offset = blockIdx.x * nums_per_block;
    int local_sum = 0;
    for (int i = threadIdx.x; i < nums_per_block && (offset + i) < num_elems; i += blockDim.x) {
        local_sum += input[offset + i];
    }
    __syncthreads();
    local_sum = block_all_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = local_sum;
    }
}






// input, output are device pointers (i.e. pointers to memory on the GPU)
void subarray_sum_cuda(const int* input, int* output, int N, int S, int E) {
    // Calculate the number of elements to sum
    const int* input_ptr = input + S;
    int num_elems = E - S + 1;
    // 退化成一维数组的reduce sum
    if (num_elems <= 1024 * 1024) {
        short_vector_reduce_sum_kernel<<<1, 1024>>>(input_ptr, output, num_elems);
        cudaDeviceSynchronize();
    }
    else {
        int grid_size = 1024;
        int nums_per_block = div_up(num_elems, grid_size);
        int *d_buf;
        CHECK_CUDA_ERROR(cudaMalloc(&d_buf, sizeof(int) * grid_size));
        vector_reduce_sum_kernel<<<grid_size, 1024>>>(input_ptr, d_buf, num_elems, nums_per_block);
        short_vector_reduce_sum_kernel<<<1, 1024>>>(d_buf, output, grid_size);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaFree(d_buf));
    }

    return;
}


torch::Tensor subarray_sum(
    const torch::Tensor& input,
    const int S, const int E
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) >= S && input.size(0) >= E, "Invalid range");
    TORCH_CHECK(S < E, "Invalid range");

    int n = input.size(0);
    torch::Tensor output = torch::zeros({1}, input.options());
    subarray_sum_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, S, E
    );
    return output;
}