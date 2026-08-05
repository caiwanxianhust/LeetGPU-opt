#include "utils.cuh"

#include <torch/extension.h>


/*
Implement a radix sort algorithm that sorts an array of 32-bit unsigned integers on a GPU. The program should take an input array of unsigned integers and sort them in ascending order using the radix sort algorithm. The input parameter contains the unsorted array, and the sorted result should be stored in the output array.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final sorted result must be stored in the output array
Use radix sort algorithm (not other sorting algorithms)
Sort in ascending order
Example 1:
  Input:  [170, 45, 75, 90, 2, 802, 24, 66]
  Output: [2, 24, 45, 66, 75, 90, 170, 802]
  
Example 2:
  Input:  [1, 4, 1, 3, 555, 1000, 2]
  Output: [1, 1, 2, 3, 4, 555, 1000]
  
Constraints
1 ≤ N ≤ 100,000,000
0 ≤ input[i] ≤ 4,294,967,295 (32-bit unsigned integers)
*/

namespace {
    const int BLOCK_SIZE = 256;
    const int RADIX      = 256;
}


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


/*
block (BLOCK_SIZE)
grid (div_up(N, BLOCK_SIZE))
*/
__global__ void radix_histgram(const unsigned int* arr, int* block_hists, int N, int shift) { 
    __shared__ int s_hist[RADIX];
    s_hist[threadIdx.x] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int digit = (arr[tid] >> shift) & 0xFFu;
        atomicAdd(&s_hist[digit], 1);
    }
    __syncthreads();

    block_hists[blockIdx.x * RADIX + threadIdx.x] = s_hist[threadIdx.x];
}


__global__ void offset_calc(const int* block_hists, int* digit_base, int* block_offsets, int num_blocks) {
    __shared__ int s_hist[RADIX];

    if (blockIdx.x == 0) {
        int disgit_counts = 0;
        for (int row = 0; row < num_blocks; ++row) {
            disgit_counts += block_hists[row * RADIX + threadIdx.x];
        }
        __syncthreads();
        s_hist[threadIdx.x] = disgit_counts;
        __syncthreads();

        if (threadIdx.x == 0) {
            int offset = 0;
            for (int i = 0; i < RADIX; ++i) {
                digit_base[i] = offset;
                offset += s_hist[i];
            }
        }
        __syncthreads();

        int idx_offset = 0;
        for (int row = 0; row < num_blocks; ++row) {
            block_offsets[row * RADIX + threadIdx.x] = idx_offset;
            idx_offset += block_hists[row * RADIX + threadIdx.x];
        }
    }
}


__global__ void radix_scatter(const unsigned int* in, unsigned int* out, const int* digit_base, const int* block_offsets,
    int N, int shift)
{ 
    __shared__ int s_base_pos[RADIX];
    __shared__ int s_local_pos[RADIX];

    for (int i = threadIdx.x; i < RADIX; i += blockDim.x) {
        s_base_pos[i] = digit_base[i] + block_offsets[blockIdx.x * RADIX + i];
        s_local_pos[i] = 0;
    }
    __syncthreads();

    unsigned int myval, mydig;
    int pos;

    if (threadIdx.x == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            int idx = blockIdx.x * blockDim.x + i;
            if (idx < N) {
                myval = in[blockIdx.x * blockDim.x + i];
                mydig = (myval >> shift) & 0xFFu;
                pos = s_base_pos[mydig] + s_local_pos[mydig];
                out[pos] = myval;
                s_local_pos[mydig]++;
            }
            
        }
    }
}



void radix_sort_cuda(const unsigned int* d_input, unsigned int* d_output, int N) {
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(unsigned int));
    cudaMalloc(&d_out, N * sizeof(unsigned int));
    cudaMemcpy(d_in, d_input, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    int threads = BLOCK_SIZE;
    int grid    = div_up(N, BLOCK_SIZE);

    int *d_blockHists, *d_blockOffs, *d_digitBase;
    cudaMalloc(&d_blockHists,  grid * RADIX * sizeof(int));
    cudaMalloc(&d_blockOffs,   grid * RADIX * sizeof(int));
    cudaMalloc(&d_digitBase,   RADIX * sizeof(int));

    for (int shift = 0; shift < 32; shift += 8) {
        radix_histgram<<<grid, threads>>>(d_in, d_blockHists, N, shift);
        offset_calc<<<1, RADIX>>>(d_blockHists, d_digitBase, d_blockOffs, grid);
        radix_scatter<<<grid, threads>>>(d_in, d_out, d_digitBase, d_blockOffs, N, shift);

        unsigned int* tmp = d_in; d_in = d_out; d_out = tmp;
    }

    cudaMemcpy(d_output, d_in, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_blockHists);
    cudaFree(d_blockOffs);
    cudaFree(d_digitBase);
}


torch::Tensor radix_sort(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kUInt32, "Input must be of type uint32");

    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);

    radix_sort_cuda(
        input.data_ptr<unsigned int>(),
        output.data_ptr<unsigned int>(),
        N
    );
    return output;
}