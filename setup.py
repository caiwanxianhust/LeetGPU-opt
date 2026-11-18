from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob


def _adaptive_jobs():
    """按可用内存选 MAX_JOBS，避免历史出现过的 nvcc OOM。

    实测（RTX 4060 Laptop / 12 核 / 15.4 GiB 主机），每个 .cu 的 nvcc 峰值 RSS：
        src/gelu.cu                 3102 MiB, 61 s
        src/fused_ln_linear_gelu.cu 3103 MiB, 67 s
        src/bindings.cu             3430 MiB, 67 s
    峰值几乎与 kernel 大小无关（大头是 torch/extension.h），故按最重值 3450 MiB/进程
    留 1% 余量折算。显式设置 MAX_JOBS 时一律以环境变量为准。
    """
    if os.environ.get("MAX_JOBS"):
        return int(os.environ["MAX_JOBS"]), "from MAX_JOBS env"
    per_job_mb = int(os.environ.get("LEETGPU_MEM_PER_JOB_MB", "3450"))
    reserve_mb = int(os.environ.get("LEETGPU_MEM_RESERVE_MB", "2048"))
    avail_mb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
                    break
    except OSError:
        pass
    if avail_mb is None:
        return 1, "MemAvailable unknown -> serial"
    cores = os.cpu_count() or 1
    jobs = max(1, min(cores, int((avail_mb - reserve_mb) // per_job_mb)))
    return jobs, f"{avail_mb} MiB avail, {cores} cores, ~{per_job_mb} MiB/compile"


_jobs, _why = _adaptive_jobs()
os.environ.setdefault("MAX_JOBS", str(_jobs))
print(f"[leetgpu] MAX_JOBS={os.environ['MAX_JOBS']} ({_why})")

setup(
    name="leetgpu",
    ext_modules=[
        CUDAExtension(
            name="leetgpu",
            sources = glob.glob("src/*.cu"),
            # sources=[
                # "src/vector_addition.cu", 
                # "src/matrix_multiplication.cu", 
                # "src/conv1D.cu",
                # "src/reverse_array.cu",
                # "src/transpose.cu",
                # "src/color_inversion.cu",
                # "src/relu.cu",
                # "src/leaky_relu.cu",
                # "src/rainbow_table.cu",
                # "src/matrix_copy.cu",
                # "src/count_array_element.cu",
                # "src/count_array_element_2D.cu",
                # "src/silu.cu",
                # "src/swiglu.cu",
                # "src/reduction.cu",
                # "src/softmax.cu",
                # "src/softmax_attention.cu",
                # "src/conv2D.cu",
                # "src/histogramming.cu",
                # "src/prefix_sum.cu",
                # "src/dot_product.cu",  # 添加 dot_product.cu
                # "src/sparse_matrix_vector_multiplication.cu",
                # "src/gemm.cu",
                # "src/categorical_cross_entropy_loss.cu",
                # "src/mean_squared_error.cu",
                # "src/gaussian_blur.cu",  # 添加 gaussian_blur.cu
                # "src/topk_selection.cu",  # 添加 topk_selection.cu
                # "src/batched_matrix_multiplication.cu",  # 添加 batched_matrix_multiplication.cu
                # "src/int8_quantized_matmul.cu",
                # "src/ordinary_least_squares.cu",  # 添加 ordinary_least_squares.cu
                # "src/logistic_regression.cu",
                # "src/monte_carlo_integration.cu",
                # "src/sorting.cu",
                # "src/radix_sort.cu",  # 添加 radix_sort.cu
                # "src/matrix_power.cu",  # 添加 matrix_power.cu
                # "src/nearest_neighbor.cu",  # 添加 nearest_neighbor.cu
                # "src/batch_norm.cu",  # 添加 batch_norm.cu
                # "src/max_pooling_2D.cu", # 添加 max_pooling_2D.cu
                # "src/count_array_element_3D.cu",  # 添加 count_array_element_3D.cu
                # "src/subarray_sum.cu",
                # "src/subarray_sum_2D.cu",
                # "src/subarray_sum_3D.cu",
                # "src/rms_norm.cu",
                # "src/max_subarray_sum.cu",
                # "src/attention_with_linear_bias.cu",
                # "src/batched_matrix_mul_fp16.cu",
                # "src/dot_product_fp16.cu",
                # "src/rope_embedding.cu",
                # "src/conv3D.cu",
                # "src/mha.cu",
                # "src/multi_agent_simulation.cu",
                # "src/kmeans_clustering.cu",
                # "src/causal_self_attention.cu",
                # "src/linear_self_attention.cu",
                # "src/sliding_window_self_attention.cu",
                # "src/weight_dequantization.cu",
                # "src/jacobi_stencil_2D.cu",
                # "src/stream_compaction.cu",
                # "src/moe_topk_gating.cu",
                # "src/gpt2_transformer_block.cu",
                # "src/all_pairs_shortest_paths.cu",
                # "src/grouped_query_attention.cu",
                # "src/linear_recurrence.cu",
                # "src/lora_linear.cu",
                # "src/int4_weight_only_quantized_matMul.cu",
                # "src/topp_sampling.cu",
                # "src/fft_2D.cu",
                # "src/fast_fft.cu",  # 添加 fast_fft.cu
                # "src/int8_kvcache_attntion.cu",
                # "src/swiglu_mlp_block.cu",
                # "src/causal_depthwise_conv1D.cu",
                # "src/llama_transformer_block.cu",
                # "src/ssm_selective_scan.cu",  # 添加 ssm_selective_scan.cu
                # "src/group_normalization.cu",  # 添加 group_normalization.cu
                # "src/token_embedding_layer.cu",
                # "src/attention_with_sinks.cu",
                # "src/grpo_surrogate_loss.cu",
                # "src/ppo_clipped_surrogate_loss.cu",
                # "src/dpo_sequence_loss.cu",  # 添加 dpo_sequence_loss.cu
                # "src/softmax_attention_backward.cu",
                # "src/adder_transformer_inference.cu",
                # "src/multi_head_cross_attention.cu",
                # "src/bindings.cu"],

            # 【新增】在这里指定 CUTLASS 的头文件目录
            include_dirs=[
                "/usr/local/cutlass/include",
                "/usr/local/cutlass/tools/util/include",
                "/usr/local/cutlass/examples/common",  # 如果代码中用到了 helper.h 等
            ],
            
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