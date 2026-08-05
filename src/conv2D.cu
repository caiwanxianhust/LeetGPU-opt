#include "utils.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_coord.h>
#include <cutlass/matrix_coord.h>
#include <torch/extension.h>


/*
Write a program that performs a 2D convolution operation on the GPU. Given an input matrix and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of:

input: A 2D matrix of 32-bit floating-point numbers, represented as a 1D array in row-major order.
kernel: A 2D kernel (filter) of 32-bit floating-point numbers, also represented as a 1D array in row-major order.
The output should be written to the output matrix (also a 1D array in row-major order). The output matrix will have dimensions:

output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output

Constraints
1 ≤ input_rows, input_cols ≤ 3072
1 ≤ kernel_rows, kernel_cols ≤ 31
kernel_rows ≤ input_rows
kernel_cols ≤ input_cols
*/



namespace naive_conv2d {

__device__ float matrix_inner_product(const float* input, const float* kernel,
                                 int rows, int cols, int stride) {
    float result = 0.0f;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result += input[i * stride + j] * kernel[i * cols + j];
        }
    }
    return result;
}


__global__ void convolution_2d_kernel(const float* input, const float* kernel, float* output,
                                      int input_rows, int input_cols, int kernel_rows, int kernel_cols) { 
    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    const float *input_ptr = input + row * input_cols + col;
    if (row < output_rows && col < output_cols) { 
        output[row * output_cols + col] = matrix_inner_product(input_ptr, kernel, kernel_rows, kernel_cols, input_cols);
    }
}


// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void convolution_2d_naive_cuda(const float* input, const float* kernel, float* output, 
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    printf("launch convolution_2d_naive_cuda!\n");

    const int output_rows = input_rows - kernel_rows + 1;
    const int output_cols = input_cols - kernel_cols + 1;
    dim3 block(16, 16);
    dim3 grid((output_cols + block.x - 1) / block.x, (output_rows + block.y - 1) / block.y);
    convolution_2d_kernel<<<grid, block>>>(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


}   // namespace naive_conv2d


namespace cutlass_conv2d {

void conv2d_simt(
    const float* input,
    const float* kernel,
    float* output,
    int n, int c, int h, int w,
    int k, int r, int s,
    int pad_h, int pad_w,
    int stride_h, int stride_w
) {
    // ========================================================================
    // 1. 使用 FP32 作为输入，FP32 作为输出和累加器
    // ========================================================================
    using ElementInput = float; // FP32 输入
    using ElementOutput = float;              // FP32 输出
    using ElementAccumulator = float;         // FP32 累加器
    using ElementComputeEpilogue = float;      // FP32 计算类型

    using LayoutInput = cutlass::layout::TensorNHWC;
    using LayoutWeight = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassSimt; // 使用 CUDA Core

    // This code section describes the tile size a thread block will compute
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;  // Threadblock tile shape

    // This code section describes tile size a warp will compute
    using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;         // Warp tile shape

    // This code section describes the size of MMA op
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;    // CUDA Core instruction shape

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

    // This code section describes the epilogue part of the kernel, we use default value
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                     // Data type of output matrix.
        1,                                                 // The number of elements per vectorized.
                                                        // memory access. This becomes the vector width of
                                                        // math instructions in the epilogue too.
        ElementAccumulator,                                // Data type of accumulator
        ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    // ========================================================================
    // 2. 定义 FP32 Tensor Core 风格的 Conv2d 操作
    // ========================================================================
    using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementInput, LayoutInput,
        ElementInput, LayoutWeight,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        MMAOp,
        cutlass::arch::Sm80,                  // Ampere 架构
        ThreadblockShape, // Threadblock Shape 
        WarpShape,   // Warp Shape
        InstructionShape,    // Instruction Shape 
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages, // Stages
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel;

    using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

    // ========================================================================
    // 3. 定义问题尺寸 (Valid Padding)
    // ========================================================================
    int out_h = (h + 2 * pad_h - r) / stride_h + 1;
    int out_w = (w + 2 * pad_w - s) / stride_w + 1;

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
    int split_k_slices = 1;

    cutlass::Tensor4DCoord input_size(n, h, w, c);
    cutlass::Tensor4DCoord filter_size(k, r, s, c);
    cutlass::Tensor4DCoord padding(pad_h, pad_h, pad_w, pad_w);
    cutlass::MatrixCoord stride(stride_h, stride_w);
    cutlass::MatrixCoord dilation(1, 1);
    cutlass::Tensor4DCoord output_size(n, out_h, out_w, k);

    cutlass::conv::Conv2dProblemSize problem_size(
        input_size,
        filter_size,
        padding,
        stride,
        dilation,
        output_size,
        mode,
        split_k_slices
    );

    // ========================================================================
    // 4. 创建 Tensor Refs (关键：使用 const_cast 移除 const 属性)
    // ========================================================================
    cutlass::TensorRef<ElementInput, LayoutInput> input_ref(
        const_cast<ElementInput*>(input), 
        LayoutInput::packed({n, h, w, c})
    );
    
    cutlass::TensorRef<ElementInput, LayoutWeight> weight_ref(
        const_cast<ElementInput*>(kernel), 
        LayoutWeight::packed({k, r, s, c})
    );

    cutlass::TensorRef<ElementOutput, LayoutOutput> output_ref(
        output, 
        LayoutOutput::packed({n, out_h, out_w, k})
    );

    // ========================================================================
    // 5. 构造 Arguments 并执行
    // ========================================================================
    Conv2dFprop implicit_gemm_op;

    typename Conv2dFprop::Arguments arguments{
        problem_size,
        input_ref,
        weight_ref,
        output_ref,
        output_ref, // source tensor
        {1.0f, 0.0f} // epilogue params (alpha, beta)
    };

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = implicit_gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS Conv2d TF32 init failed!" << std::endl;
        return;
    }

    status = implicit_gemm_op();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS Conv2d TF32 run failed!" << std::endl;
    }
}

// input, kernel, output are device pointers
void conv2d_cutlass_cuda(const float* input, const float* kernel, float* output, 
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    printf("launch conv2d_cutlass_cuda!\n");
    
    // 1. 设定合理的默认参数（单通道，无 padding，步长为 1）
    int n = 1;            
    int c = 1;            
    int k = 1;            
    int pad_h = 0;        
    int pad_w = 0;        
    int stride_h = 1;     
    int stride_w = 1;     

    // 2. 映射 2D 尺寸到 4D 卷积参数
    int h = input_rows;
    int w = input_cols;
    int r = kernel_rows;
    int s = kernel_cols;

    // 3. 调用 CUTLASS SIMT Conv2d 实现
    conv2d_simt(
        input, 
        kernel, 
        output, 
        n, c, h, w, 
        k, r, s, 
        pad_h, pad_w, 
        stride_h, stride_w
    );
}

}   // namespace cutlass_conv2d



void convolution_2d_cuda(const float* input, const float* kernel, float* output, 
    int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    // naive_conv2d::convolution_2d_naive_cuda(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
    cutlass_conv2d::conv2d_cutlass_cuda(input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols);
}


torch::Tensor convolution_2D(
    const torch::Tensor& input,
    const torch::Tensor& kernel,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(kernel.ndimension() == 1, "Kernel must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(kernel.device().is_cuda(), "Kernel must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be of type float32");
    TORCH_CHECK(input.size(0) >= kernel.size(0), "Input must be longer than kernel");
    TORCH_CHECK(input.size(0) == input_rows * input_cols, "Input must be a 1D matrix");
    TORCH_CHECK(kernel.size(0) == kernel_rows * kernel_cols, "Kernel must be a 1D matrix");

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    auto options = input.options();
    torch::Tensor output = torch::empty({output_rows * output_cols}, options);
    convolution_2d_cuda(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        input_rows, input_cols, kernel_rows, kernel_cols
    );
    return output;
}