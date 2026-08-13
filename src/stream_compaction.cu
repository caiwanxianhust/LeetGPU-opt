#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Given a 1D array A of N 32-bit floating point numbers, compact all positive elements (A[i] > 0) to the front of the output array out, preserving their original relative order. Fill any remaining positions with 0.0. Stream compaction is a fundamental GPU primitive used throughout rendering, sparse computation, and collision detection.

Implementation Requirements
Use only native GPU features (external libraries are not permitted)
The solve function signature must remain unchanged
The first k positions of out must contain the k elements of A where A[i] > 0, in their original order
Positions k through N−1 of out must be 0.0
Elements where A[i] = 0.0 are not selected
Example
Input:  A = [1.0, -2.0, 3.0, 0.0, -1.0, 4.0]
Output: out = [1.0, 3.0, 4.0, 0.0, 0.0, 0.0]
Constraints
1 ≤ N ≤ 100,000,000
−1000.0 ≤ A[i] ≤ 1000.0
out is pre-allocated with N elements, initialised to 0.0
Performance is measured with N = 50,000,000
*/



__global__ void block_count_kernel(const float* A, int N, int* d_block_cnt, int nums_per_block) {
    int offset = blockIdx.x * nums_per_block;
    int cnt = 0;
    for (int i = threadIdx.x; i < nums_per_block && offset + i < N; i += blockDim.x) {
        cnt += (A[offset + i] > 0) ? 1 : 0;
    }
    __syncthreads();

    cnt = block_all_reduce_sum(cnt);
    if (threadIdx.x == 0) {
        d_block_cnt[blockIdx.x] = cnt;
    }
}

__global__ void scan_kernel(const int* block_count, int N, int* idx_off) { 
    __shared__ int s_block_count[1024];
    int tid = threadIdx.x;
    s_block_count[tid] = (tid < N) ? block_count[tid] : 0;
    __syncthreads();

    int val = 0;
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        if (tid >= offset) {
            val = s_block_count[tid] + s_block_count[tid - offset];
        }
        __syncthreads();
        if (tid >= offset) {
            s_block_count[tid] = val;
        }
    }
    if (tid < N) {
        idx_off[tid] = (tid == 0) ? 0 : s_block_count[tid - 1];
    }
}

__global__ void stream_compaction_kernel(const float* A, int N, float* out, int* idx_off, int nums_per_block) {
    int offset = threadIdx.x * nums_per_block;
    int cur_idx_offset = idx_off[threadIdx.x];
    for (int i = 0; i < nums_per_block && offset + i < N; i++) {
        if (A[offset + i] > 0.0f) {
            out[cur_idx_offset] = A[offset + i];
            cur_idx_offset++;
        }
    }
}

// A, out are device pointers
void stream_compaction_cuda(const float* A, int N, float* out) 
{

    int block_size = 1024;
    int grid_size;
    if (N <= block_size * 1024) {
        grid_size = div_up(N, block_size);
    }
    else {
        grid_size = 1024;
    }
    int nums_per_block = div_up(N, grid_size);

    int *d_block_cnt;
    int *d_idx_off;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_block_cnt, sizeof(int) * 1024 * 2));
    d_idx_off = d_block_cnt + 1024;
    // 计算每个 block 中非零元素的个数
    block_count_kernel<<<grid_size, block_size>>>(A, N, d_block_cnt, nums_per_block);
    // 计算非零元素在 out 中的偏移
    scan_kernel<<<1, 1024>>>(d_block_cnt, grid_size, d_idx_off);
    // 拷贝非零元素
    stream_compaction_kernel<<<1, 1024>>>(A, N, out, d_idx_off, nums_per_block);
}


torch::Tensor stream_compaction(const torch::Tensor& input)
{
    int N = input.size(0);
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    torch::Tensor output = torch::zeros_like(input);
    stream_compaction_cuda(input.data_ptr<float>(), N, output.data_ptr<float>());
    return output;
}