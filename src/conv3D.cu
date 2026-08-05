#include "utils.cuh"

#include <cuda.h>
#include <torch/extension.h>

/*
Implement a program that performs a 3D convolution operation. Given a 3D input volume and a 3D kernel (filter), compute the convolved output. The convolution should use a "valid" boundary condition (no padding).

For a 3D convolution, the output at position 
 is given by:


The input consists of:

input: A 3D volume of 32-bit floats, as a 1D array (row-major, then depth).
kernel: A 3D kernel of 32-bit floats, as a 1D array (row-major, then depth).
input_depth, input_rows, input_cols: Dimensions of the input.
kernel_depth, kernel_rows, kernel_cols: Dimensions of the kernel.
Output:
output: A 1D array (row-major, then depth) storing the result.
Output dimensions:
output_depth = input_depth - kernel_depth + 1
output_rows = input_rows - kernel_rows + 1
output_cols = input_cols - kernel_cols + 1
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in output


Constraints
1 ≤ input_depth, input_rows, input_cols ≤ 256
1 ≤ kernel_depth, kernel_rows, kernel_cols ≤ 5
kernel_depth ≤ input_depth
kernel_rows ≤ input_rows
kernel_cols ≤ input_cols
*/

static __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void conv3D_kernel(const float* input, const float* kernel, float* output,
                              int input_depth, int input_rows, int input_cols,
                              int kernel_depth, int kernel_rows, int kernel_cols,
                              int output_depth, int output_rows, int output_cols) 
{
    // Calculate the global thread indices
    int odepth = blockIdx.z; // output depth index
    int orow = blockIdx.y * blockDim.y + threadIdx.y; // output row index
    int ocol = blockIdx.x * blockDim.x + threadIdx.x; // output column index

    // Check if the thread is within the bounds of the output volume
    if (odepth < output_depth && orow < output_rows && ocol < output_cols) {
        float sum = 0.0f;

        // Perform the convolution operation
        for (int kd = 0; kd < kernel_depth; ++kd) {
            for (int kr = 0; kr < kernel_rows; ++kr) {
                for (int kc = 0; kc < kernel_cols; ++kc) {
                    int id = odepth + kd;
                    int ir = orow + kr;
                    int ic = ocol + kc;

                    sum += input[(id * input_rows * input_cols) + (ir * input_cols) + ic] *
                           kernel[(kd * kernel_rows * kernel_cols) + (kr * kernel_cols) + kc];
                }
            }
        }

        // Store the result in the output volume
        output[(odepth * output_rows * output_cols) + (orow * output_cols) + ocol] = sum;
    }
}




void conv3D_cuda(const float* input, const float* kernel, float* output, int input_depth,
                      int input_rows, int input_cols, int kernel_depth, int kernel_rows,
                      int kernel_cols) 
{
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    // Define block and grid sizes
    constexpr int TILE_DIM = 16;
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(div_up(output_cols, TILE_DIM), div_up(output_rows, TILE_DIM), output_depth);

    conv3D_kernel<<<grid, block>>>(input, kernel, output, input_depth, input_rows, input_cols,
                                   kernel_depth, kernel_rows, kernel_cols,
                                   output_depth, output_rows, output_cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}


torch::Tensor conv3D(
    const torch::Tensor& input,
    const torch::Tensor& kernel
) { 
    TORCH_CHECK(input.ndimension() == 3, "Input must be 3D");
    TORCH_CHECK(kernel.ndimension() == 3, "Kernel must be 3D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(kernel.device().is_cuda(), "Kernel must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be of type float32");

    int input_depth = input.size(0);
    int input_rows = input.size(1);
    int input_cols = input.size(2);
    int kernel_depth = kernel.size(0);
    int kernel_rows = kernel.size(1);
    int kernel_cols = kernel.size(2);

    TORCH_CHECK(kernel_depth <= input_depth && kernel_rows <= input_rows && kernel_cols <= input_cols, 
        "Kernel dimensions must be less than or equal to input dimensions");
    
    int output_depth = input_depth - kernel_depth + 1;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;
    torch::Tensor output = torch::empty({output_depth, output_rows, output_cols}, input.options());
    conv3D_cuda(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols
    );

    return output;
}