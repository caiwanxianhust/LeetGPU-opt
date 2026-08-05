#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>
#include <torch/extension.h>


/*
Implement a GPU program that, for N three-dimensional points stored on the device, fills indices[i] with the index j ≠ i of the point closest to points[i]. Comparing squared Euclidean distance is sufficient—you do not need to compute square-roots.

Implementation Requirements
The solve function signature must remain unchanged
External libraries are not permitted
The final result must be stored in the indices array
Example 1:
Input:  points  = [(0,0,0), (1,0,0), (5,5,5)]
        indices = [-1, -1, -1]
        N       = 3
Output: indices = [1, 0, 1]   # 0⇆1 are nearest, 2 is closest to 1
Constraints
1 ≤ N ≤ 100,000
Coordinates are 32-bit floats in the range [-1000, 1000]
*/

namespace {
    constexpr int BLOCK_SIZE = 256;
}

struct VAL_INDEX {
    float val;
    int index;
};

static __device__ VAL_INDEX warp_reduce_min(VAL_INDEX val_index) {
    VAL_INDEX other;
    for (int offset = 16; offset > 0; offset /= 2) {
        other.val = __shfl_xor_sync(0xffffffff, val_index.val, offset);
        other.index = __shfl_xor_sync(0xffffffff, val_index.index, offset);
        if (other.val < val_index.val) {
            val_index = other;
        }
    }
    return val_index;
}

static __device__ VAL_INDEX block_reduce_min(VAL_INDEX val_index) { 
    val_index = warp_reduce_min(val_index);
    __shared__ VAL_INDEX shared[32]; // BLOCK_SIZE / warpSize
    __shared__ VAL_INDEX ret;
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane_id == 0) {
        shared[warp_id] = val_index;
    }
    __syncthreads();
    
    val_index = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : VAL_INDEX{FLT_MAX, -1};
    val_index = warp_reduce_min(val_index);
    if (threadIdx.x == 0) {
        ret = val_index;
    }
    __syncthreads();
    return ret;
}


static __global__ void nearest_neighbor_kernel(const float* points, int* indices, int N) {
    const float3 *points_ptr = reinterpret_cast<const float3*>(points);
    int idx_cur = blockIdx.x;
    float3 p = points_ptr[idx_cur];

    __shared__ float min_dists[BLOCK_SIZE];
    __shared__ int min_indexs[BLOCK_SIZE];
    min_dists[threadIdx.x] = FLT_MAX;
    min_indexs[threadIdx.x] = -1;
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (i != idx_cur) {
            float3 q = points_ptr[i];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dz = p.z - q.z;
            float dist = dx * dx + dy * dy + dz * dz;

            if (dist < min_dists[threadIdx.x]) {
                min_dists[threadIdx.x] = dist;
                min_indexs[threadIdx.x] = i;
            }
        }
    }
    __syncthreads();
    
    VAL_INDEX val_index{min_dists[threadIdx.x], min_indexs[threadIdx.x]};
    VAL_INDEX block_min = block_reduce_min(val_index);
    if (threadIdx.x == 0) {
        indices[idx_cur] = block_min.index;
    }
}



// points and indices are device pointers
void nearest_neighbor_cuda(const float* points, int* indices, int N) {
    nearest_neighbor_kernel<<<N, BLOCK_SIZE>>>(points, indices, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor nearest_neighbor(
    const torch::Tensor& points
) {
    TORCH_CHECK(points.ndimension() == 2, "points arr must be 2D");
    TORCH_CHECK(points.device().is_cuda(), "points must be on CUDA");
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be of type float32");
    TORCH_CHECK(points.size(1) == 3, "points must be 3D");
    int N = points.size(0);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor indices = torch::empty({N}, options);

    nearest_neighbor_cuda(
        points.data_ptr<float>(),
        indices.data_ptr<int>(),
        N
    );

    return indices;
}
