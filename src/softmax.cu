#include "utils.cuh"
#include <cfloat>


#include "cuda_kernels.cuh"

#include <torch/extension.h>



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


torch::Tensor softmax_1D(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    torch::Tensor output = torch::empty({N}, input.options());
    softmax_1D_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}
