from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ['MAX_JOBS'] = '1'  # 限制为单进程编译

setup(
    name="leetgpu",
    ext_modules=[
        CUDAExtension(
            name="leetgpu",
            sources=[
                "src/vector_addition.cu", 
                "src/matrix_multiplication.cu", 
                "src/conv1D.cu",
                "src/reverse_array.cu",
                "src/transpose.cu",
                "src/color_inversion.cu",
                "src/relu.cu",
                "src/leaky_relu.cu",
                "src/rainbow_table.cu",
                "src/matrix_copy.cu",
                "src/count_array_element.cu",
                "src/count_array_element_2D.cu",
                "src/silu.cu",
                "src/swiglu.cu",
                "src/reduction.cu",
                "src/softmax.cu",
                "src/softmax_attention.cu",
                "src/conv2D.cu",
                "src/histogramming.cu",
                "src/prefix_sum.cu",
                "src/dot_product.cu",  # 添加 dot_product.cu
                "src/sparse_matrix_vector_multiplication.cu",
                "src/gemm.cu",
                "src/categorical_cross_entropy_loss.cu",
                "src/mean_squared_error.cu",
                "src/gaussian_blur.cu",  # 添加 gaussian_blur.cu
                "src/topk_selection.cu",  # 添加 topk_selection.cu
                "src/batched_matrix_multiplication.cu",  # 添加 batched_matrix_multiplication.cu
                "src/int8_quantized_matmul.cu",
                "src/ordinary_least_squares.cu",  # 添加 ordinary_least_squares.cu
                "src/logistic_regression.cu",
                "src/monte_carlo_integration.cu",
                "src/sorting.cu",
                "src/radix_sort.cu",  # 添加 radix_sort.cu
                "src/matrix_power.cu",  # 添加 matrix_power.cu
                "src/nearest_neighbor.cu",  # 添加 nearest_neighbor.cu
                "src/batch_norm.cu",  # 添加 batch_norm.cu
                "src/max_pooling_2D.cu", # 添加 max_pooling_2D.cu
                "src/count_array_element_3D.cu",  # 添加 count_array_element_3D.cu
                "src/subarray_sum.cu",
                "src/subarray_sum_2D.cu",
                "src/subarray_sum_3D.cu",
                "src/rms_norm.cu",
                "src/max_subarray_sum.cu",
                "src/attention_with_linear_bias.cu",
                "src/batched_matrix_mul_fp16.cu",
                "src/dot_product_fp16.cu",
                "src/rope_embedding.cu",
                "src/conv3D.cu",
                "src/mha.cu",
                "src/multi_agent_simulation.cu",
                "src/bindings.cu"],
            extra_compile_args={
                "nvcc": [
                    # "-O0",  # 降低优化级别减少内存占用
                    "-arch=sm_89",  # 替换为你的GPU架构（如sm_61, sm_86）
                    "-std=c++17"
                ],
                "cxx": [
                    # "-O0", 
                    "-std=c++17"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)

# python3 setup.py build_ext --inplace