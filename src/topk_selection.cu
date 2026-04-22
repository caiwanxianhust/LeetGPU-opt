#include "utils.cuh"

#include <cuda.h>
#include <cfloat>
#include <assert.h>


/*
Implement a GPU program that, given a 1D array input of 32-bit floating point numbers of length N, selects the k largest elements and writes them in descending order to the output array of length k.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the output array
Example 1:
  Input:
  input = [1.0, 5.0, 3.0, 2.0, 4.0]
  N = 5
  k = 3
  
  Output:
  output = [5.0, 4.0, 3.0]
  
Example 2:
  Input:
  input = [7.2, -1.0, 3.3, 8.8, 2.2]
  N = 5
  k = 2
  
  Output:
  output = [8.8, 7.2]
  
Constraints
1 ≤ N ≤ 100,000,000
1 ≤ k ≤ N
All values in input are 32-bit floats
*/



// __device__ __forceinline__ void inset_to_min_heap(float *heap, float val, int k) {
//     if (val > heap[k - 1]) {
//         heap[k - 1] = val;
//         for (int i = k - 1; i > 0; i--) {
//             if (heap[i] > heap[i - 1]) {
//                 float temp = heap[i];
//                 heap[i] = heap[i - 1];
//                 heap[i - 1] = temp;
//             }
//         }
//     }
// }

__device__ __forceinline__ void inset_to_min_heap(float *heap, float val, int k) {
    if (val < heap[k - 1]) return;
    int i = k - 2;
    while (i >= 0 && heap[i] < val) {
        heap[i + 1] = heap[i];
        i--;
    }
    heap[i + 1] = val;
}


__device__ __forceinline__ void combine_min_heaps(float *heap1, float *heap2, int k) {
    for (int i = 0; i < k; i++) {
        inset_to_min_heap(heap1, heap2[i], k);
    }
}



__global__ void topk_in_block_kernel(
    const float* input, float* block_topk, int N, int k
) {
    extern __shared__ float shared_topk[];
    float *s_min_heap = shared_topk + threadIdx.x * k;

    // Initialize the min-heap
    for (int i = 0; i < k; i++) {
        s_min_heap[i] = -FLT_MAX;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        inset_to_min_heap(s_min_heap, input[i], k);
    }
    __syncthreads();

    // block-wide reduction to find the k largest elements
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            combine_min_heaps(s_min_heap, s_min_heap + offset * k, k);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            block_topk[blockIdx.x * k + i] = s_min_heap[i];
        }
    }
}

void small_k_topk_cuda(
    const float* input, float* output, int N, int k
) {
    int block_size = 128;
    int grid_size;
    if (N < block_size) {
        grid_size = 1;
    } 
    else if (N < 128 * block_size) {
        grid_size = (N + block_size - 1) / block_size;
    }
    else {
        grid_size = 128;
    }
    
    int shared_mem_size = block_size * k * sizeof(float);
    float *d_block_topk;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_topk, grid_size * k * sizeof(float)));
    topk_in_block_kernel<<<grid_size, block_size, shared_mem_size, 0>>>(input, d_block_topk, N, k);
    topk_in_block_kernel<<<1, grid_size, shared_mem_size, 0>>>(d_block_topk, output, grid_size * k, k);
    CHECK_CUDA_ERROR(cudaFree(d_block_topk));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void big_k_topk_in_block_kernel(
    const float* input, float* output, float* d_block_topk, int N, int k
) {
    // Each thread has its own min-heap in global memory
    float* min_heap = d_block_topk + threadIdx.x * k + blockIdx.x * k * blockDim.x;
    // Initialize the min-heap
    for (int i = 0; i < k; i++) {
        min_heap[i] = -FLT_MAX;
    }

    // Each thread processes a portion of the input array
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        inset_to_min_heap(min_heap, input[i], k);
    }
    __syncthreads();

    // block-wide reduction to find the k largest elements
    // Combine the min-heaps of all threads in the block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            combine_min_heaps(min_heap, min_heap + offset * k, k);
        }
        __syncthreads();
    }
}

__global__ void min_heap_reduce_kernel(
    float* output, float* d_block_topk, int N, int k, int offset
) { 
    float* min_heap = d_block_topk + threadIdx.x * offset * k;

    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
        if (threadIdx.x < mask) {
            combine_min_heaps(min_heap, min_heap + mask * offset * k, k);
        }
        __syncthreads();
    }
    
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        output[i] = d_block_topk[i];
    }
}

void big_k_topk_cuda(
    const float* input, float* output, int N, int k
) {
    int block_size = 1024;
    int grid_size = 128;
    assert(grid_size <= block_size);
    float *d_block_topk;
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_topk, grid_size * block_size * k * sizeof(float)));
    big_k_topk_in_block_kernel<<<grid_size, block_size>>>(input, output, d_block_topk, N, k);
    min_heap_reduce_kernel<<<1, grid_size>>>(output, d_block_topk, N, k, block_size);
    CHECK_CUDA_ERROR(cudaFree(d_block_topk));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void topk_cuda(
    const float* input, float* output, int N, int k
) {
    if (k <= 32) {
        small_k_topk_cuda(input, output, N, k);
    }
    else {
        // TODO: implement large k topk
        big_k_topk_cuda(input, output, N, k);
    }
}