#include "utils.cuh"
#include <cfloat>


#include "cuda_kernels.cuh"


// struct __align__(8) MD_F
// {
//     float m;    // max val
//     float d;    // exp sum

//     __device__ __forceinline__ MD_F() {
//         m = -FLT_MAX;
//         d = 0.0f;
//     }
// };

// struct MDF_OP {
//     __device__ __forceinline__ MD_F operator()(MD_F &a, MD_F &b) {
//         MD_F ret;
//         ret.m = max(a.m, b.m);
//         ret.d = a.d * expf(a.m - ret.m) + b.d * expf(b.m - ret.m);
//         return ret;
//     }
// };


// __device__ MD_F reduce_warp_mdf(MD_F val) { 
//     MD_F b;
//     for (int mask = 16; mask > 0; mask >>= 1) {
//         b.m = __shfl_xor_sync(0xffffffff, val.m, mask);
//         b.d = __shfl_xor_sync(0xffffffff, val.d, mask);
//         val = MDF_OP()(val, b);
//     }
//     return val;
// }

// __device__ MD_F reduce_block_mdf(MD_F val) { 
//     __shared__ MD_F result;
//     __shared__ MD_F s_buf[32];

//     const int lane_id = threadIdx.x & 0x1f;
//     const int warp_id = threadIdx.x >> 5;

//     val = reduce_warp_mdf(val);
//     if (lane_id == 0) {
//         s_buf[warp_id] = val;
//     }
//     __syncthreads();

//     val = threadIdx.x < (blockDim.x >> 5) ? s_buf[lane_id] : MD_F();
//     val = reduce_warp_mdf(val);
//     if (threadIdx.x == 0) {
//         result = val;
//     }
//     __syncthreads();
//     return result;
// }


__global__ void softmax_in_block(const float* input, MD_F* md, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    MD_F tmp;
    tmp.m = tid < N ? input[tid] : -FLT_MAX;
    tmp.d = tid < N ? 1.0f : 0.0f;
    __syncthreads();

    tmp = block_all_reduce_sum_mdf(tmp);
    if (threadIdx.x == 0) {
        md[blockIdx.x] = tmp;
    } 
}

__global__ void reduce_mdf(MD_F* md, int length) { 
    MD_F tmp;
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        tmp = MDF_Op()(tmp, md[i]);
    }
    __syncthreads();
    tmp = block_all_reduce_sum_mdf(tmp);
    if (threadIdx.x == 0) {
        md[0] = tmp;
    }
}

__global__ void softmax_kernel(const float* input, float* output, MD_F* md, int N) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float m = md[0].m;
        float d = md[0].d;
        output[tid] = expf(input[tid] - m) / d;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void softmax_1D_cuda(const float* input, float* output, int N) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    MD_F * md;
    CHECK_CUDA_ERROR(cudaMalloc(&md, sizeof(MD_F) * grid_size));
    softmax_in_block<<<grid_size, block_size>>>(input, md, N);
    reduce_mdf<<<1, 256>>>(md, grid_size);
    softmax_kernel<<<grid_size, block_size>>>(input, output, md, N);
    CHECK_CUDA_ERROR(cudaFree(md));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}