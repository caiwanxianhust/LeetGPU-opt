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