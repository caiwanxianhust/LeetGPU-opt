#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>


struct __align__(8) MD_F
{
    float m;    // max val
    float d;    // exp sum

    // __device__ __forceinline__ MD_F() {
    //     m = -FLT_MAX;
    //     d = 0.0f;
    // }
};


struct MDF_Op
{
    /* data */
    __device__ MD_F operator()(const MD_F& a, const MD_F& b) const {
        MD_F result;
        result.m = fmaxf(a.m, b.m);
        result.d = a.d * expf(a.m - result.m) + b.d * expf(b.m - result.m);
        return result;
    }
};


static __device__ float warp_all_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ int warp_all_reduce_sum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ double warp_all_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ int warp_all_reduce_max(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}


static __device__ float warp_all_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}



static __device__ MD_F warp_all_reduce_sum_mdf(MD_F val) {
    MD_F b;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        b.m = __shfl_xor_sync(0xffffffff, val.m, offset);
        b.d = __shfl_xor_sync(0xffffffff, val.d, offset);
        val = MDF_Op()(val, b);
    }
    return val;
}

static __device__ float block_all_reduce_sum(float val) {
    __shared__ float ret;
    __shared__ float shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : 0.0f;
    val = warp_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}

static __device__ int block_all_reduce_sum(int val) {
    __shared__ int ret;
    __shared__ int shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : 0;
    val = warp_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}

static __device__ double block_all_reduce_sum(double val) {
    __shared__ double ret;
    __shared__ double shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_sum(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : 0.0;
    val = warp_all_reduce_sum(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}


static __device__ MD_F block_all_reduce_sum_mdf(MD_F val) {
    __shared__ MD_F s_buf[32];
    __shared__ MD_F result;

    const int lane_id = threadIdx.x & 0x1f;
    const int warp_id = threadIdx.x >> 5;

    val = warp_all_reduce_sum_mdf(val);
    if (lane_id == 0) {
        s_buf[warp_id] = val;
    }
    __syncthreads();

    MD_F tmp;
    tmp.m = -INFINITY;
    tmp.d = 0.0f;
    val = threadIdx.x < (blockDim.x >> 5) ? s_buf[threadIdx.x] : tmp;
    val = warp_all_reduce_sum_mdf(val);
    if (threadIdx.x == 0) {
        result = val;
    }
    __syncthreads();
    return result;
}

static __device__ int block_all_reduce_max(int val) {
    __shared__ int ret;
    __shared__ int shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_max(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : -INT_MAX;
    val = warp_all_reduce_max(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}

static __device__ float block_all_reduce_max(float val) {
    __shared__ float ret;
    __shared__ float shared[32]; // Assuming max 1024 threads per block
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_all_reduce_max(val);
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[threadIdx.x] : -INFINITY;
    val = warp_all_reduce_max(val);

    if (threadIdx.x == 0) {
        ret = val;
    }
    __syncthreads();
    
    return ret;
}