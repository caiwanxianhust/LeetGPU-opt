#include <torch/extension.h>



// -------------------------- Python 绑定函数 --------------------------
torch::Tensor vector_add(const torch::Tensor& A, const torch::Tensor& B);


torch::Tensor matrix_multiplication(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const bool transpose_B = false
);


torch::Tensor convolution_1D(const torch::Tensor& input, const torch::Tensor& kernel);


void reverse_array(const torch::Tensor& input);


torch::Tensor matrix_transpose(const torch::Tensor& input);

void color_invert(const torch::Tensor& image, int width, int height);


torch::Tensor relu(const torch::Tensor& input);


torch::Tensor leaky_relu(const torch::Tensor& input);


torch::Tensor rainbow_table(const torch::Tensor& input, int R);


torch::Tensor matrix_copy(const torch::Tensor& A, int N);


torch::Tensor count_equal(const torch::Tensor& input, int K);


torch::Tensor count_equal_2D(const torch::Tensor& input, int M, int N, int K);


torch::Tensor silu(const torch::Tensor& input);


torch::Tensor swiglu(const torch::Tensor& input);


torch::Tensor reduce_sum(const torch::Tensor& input);


torch::Tensor softmax_1D(const torch::Tensor& input);


torch::Tensor softmax_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
);


torch::Tensor convolution_2D(
    const torch::Tensor& input,
    const torch::Tensor& kernel,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
);


torch::Tensor histogram(const torch::Tensor& input, int num_bins);


torch::Tensor prefix_sum(const torch::Tensor& input);


torch::Tensor dot_product(const torch::Tensor& A, const torch::Tensor& B);


torch::Tensor sparse_matrix_vector_multiplication(
    const torch::Tensor& A,
    const torch::Tensor& x,
    int M, int N, int nnz
);


torch::Tensor gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M, int N, int K,
    float alpha, float beta
);


torch::Tensor avg_categorical_cross_entropy_loss(
    const torch::Tensor& logits,
    const torch::Tensor& true_labels,
    int N, int C
);


torch::Tensor mse_vec(const torch::Tensor& predictions, const torch::Tensor& targets);


torch::Tensor gaussian_blur(const torch::Tensor& input, const torch::Tensor& kernel);


torch::Tensor topk_1D(const torch::Tensor& input, int k);


torch::Tensor batched_matmul(const torch::Tensor& A, const torch::Tensor& B);


torch::Tensor int8_quantized_matmul(
    const torch::Tensor& A,
    const torch::Tensor& B,
    float scale_A, float scale_B, float scale_C,
    int zero_point_A, int zero_point_B, int zero_point_C
);


torch::Tensor ordinary_least_squares(const torch::Tensor& X, const torch::Tensor& y);


torch::Tensor logistic_regression(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& beta);


torch::Tensor monte_carlo_integration(
    const torch::Tensor& y_samples,
    float a, float b);


void sorting(const torch::Tensor& input);


torch::Tensor radix_sort(const torch::Tensor& input);


torch::Tensor matrix_power(const torch::Tensor& input, int power);


torch::Tensor nearest_neighbor(const torch::Tensor& points);


torch::Tensor max_pooling_2D(
    const torch::Tensor& input,
    int kernel_size,
    int stride,
    int padding);


torch::Tensor batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps);


torch::Tensor count_equal_3D(
    const torch::Tensor& input, const int n, const int m, const int k,
    const int p);


torch::Tensor subarray_sum(const torch::Tensor& input, const int S, const int E);


torch::Tensor subarray_sum_2D(
    const torch::Tensor& input, const int S_ROW, const int E_ROW,
    const int S_COL, const int E_COL);


torch::Tensor subarray_sum_3D(
    const torch::Tensor& input, const int S_DEP, const int E_DEP,
    const int S_ROW, const int E_ROW, const int S_COL, const int E_COL
);


torch::Tensor rms_norm_1D(const torch::Tensor& input, float eps, float gamma, float beta);


torch::Tensor max_subarray_sum(const torch::Tensor& input, const int window_size);


torch::Tensor attention_with_linear_bias(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    float alpha);


torch::Tensor batched_matrix_mul_fp16(const torch::Tensor& A, const torch::Tensor& B);


torch::Tensor dot_product_fp16(const torch::Tensor& A, const torch::Tensor& B);


torch::Tensor rope_embedding(
    const torch::Tensor& input,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table
);


torch::Tensor conv3D(const torch::Tensor& input, const torch::Tensor& kernel);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kmeans_clustering(
    const torch::Tensor& data_x,
    const torch::Tensor& data_y,
    const torch::Tensor& init_centroids_x,
    const torch::Tensor& init_centroids_y,
    const int max_iterations);


torch::Tensor mha(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int heads);


torch::Tensor multi_agent_sim(const torch::Tensor& agents);


torch::Tensor causal_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V);


torch::Tensor linear_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V);


torch::Tensor sliding_window_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int window_size);


torch::Tensor weight_dequantization(
    const torch::Tensor& X,
    const torch::Tensor& S,
    const int tile_size);


torch::Tensor jacobi_stencil_2D(const torch::Tensor& input);


torch::Tensor stream_compaction(const torch::Tensor& input);


std::tuple<torch::Tensor, torch::Tensor> moe_topk_gating(const torch::Tensor& logits, int k);


torch::Tensor gpt2_transformer_block(const torch::Tensor& x, const torch::Tensor& weights);


torch::Tensor all_pairs_shortest_paths(const torch::Tensor& graph);


torch::Tensor gqa(const torch::Tensor& Q, const torch::Tensor& K,
    const torch::Tensor& V);


torch::Tensor linear_recurrence(const torch::Tensor& a, const torch::Tensor& x);


torch::Tensor lora_linear(
    const torch::Tensor& x,
    const torch::Tensor& W,
    const torch::Tensor& A,
    const torch::Tensor& B,
    float lora_scale);


torch::Tensor int4_weight_only_quantized_matMul(
    const torch::Tensor& x,
    const torch::Tensor& w_q,
    const torch::Tensor& scales,
    const int group_size);


torch::Tensor topp_sampling(
    const torch::Tensor& logits,
    const torch::Tensor& p,
    const torch::Tensor& seed);


torch::Tensor fft_2D(const torch::Tensor& input);


torch::Tensor fast_fft(const torch::Tensor& input);


torch::Tensor int8_kvcache_attntion(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& k_scale,
    const torch::Tensor& v_scale);


torch::Tensor swiglu_mlp_block(
    const torch::Tensor& x,
    const torch::Tensor& W_gate,
    const torch::Tensor& W_up,
    const torch::Tensor& W_down);


torch::Tensor causal_depthwise_conv1D(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias);


torch::Tensor llama_transformer_block(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& cos,
    const torch::Tensor& sin);


torch::Tensor ssm_selective_scan(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& skip);

torch::Tensor group_normalization(
    const torch::Tensor& x,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    int G, float eps);

torch::Tensor token_embedding_layer(
    const torch::Tensor& token_ids,
    const torch::Tensor& position_ids,
    const torch::Tensor& token_embeddings,
    const torch::Tensor& position_embeddings,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps);


torch::Tensor attention_with_sinks(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    int num_sinks, int window_size);


torch::Tensor grpo_surrogate_loss(
    const torch::Tensor& rewards,
    const torch::Tensor& log_pi,
    const torch::Tensor& log_pi_old,
    const torch::Tensor& log_ref,
    float clip_eps, float beta);


torch::Tensor ppo_clipped_surrogate_loss(
    const torch::Tensor& advantages,
    const torch::Tensor& log_pi,
    const torch::Tensor& log_pi_old,
    float clip_eps);


torch::Tensor dpo_sequence_loss(
    const torch::Tensor& chosen_logps, const torch::Tensor& rejected_logps,
    const torch::Tensor& chosen_ref_logps, const torch::Tensor& rejected_ref_logps,
    float beta
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> softmax_attention_backward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const torch::Tensor& dO
);


torch::Tensor adder_transformer_inference(
    const torch::Tensor& prompts,
    const torch::Tensor& weights
);


torch::Tensor multi_head_cross_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
);


torch::Tensor parallel_reverse_scan_gae(
    torch::Tensor& rewards, 
    torch::Tensor& values, 
    float gamma, float lam
); 


torch::Tensor layer_normalization(
    torch::Tensor& input, 
    torch::Tensor& weight, 
    torch::Tensor& bias, 
    float eps
);


torch::Tensor gelu(
    const torch::Tensor& X
);


torch::Tensor distillation_kl_divergence(
    const torch::Tensor& teacher_logits, const torch::Tensor& student_logits, float temperature
); 

torch::Tensor integral_image(const torch::Tensor& input);

torch::Tensor grouped_gemm(
    const torch::Tensor& A, 
    const torch::Tensor& B, 
    const torch::Tensor& group_offsets
); 

std::tuple<torch::Tensor, torch::Tensor> rms_norm_backward(
    const torch::Tensor& x, 
    const torch::Tensor& weight, 
    const torch::Tensor& dy, 
    float eps
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_backward(
    const torch::Tensor& x, 
    const torch::Tensor& gamma, 
    const torch::Tensor& mean, 
    const torch::Tensor& rstd, 
    const torch::Tensor& dy
);


torch::Tensor multi_lora_batched_linear(
    const torch::Tensor &x, 
    const torch::Tensor &W, 
    const torch::Tensor &A, 
    const torch::Tensor &B,
    const torch::Tensor &adapter_ids, 
    float scale
); 


void fused_adamw(
    torch::Tensor params,
    torch::Tensor grad,
    torch::Tensor m,
    torch::Tensor v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t
); 


torch::Tensor fused_ln_linear_gelu(
    const torch::Tensor &x, 
    const torch::Tensor &gamma, 
    const torch::Tensor &beta,
    const torch::Tensor &W, 
    const torch::Tensor &bias, 
    float eps
); 


torch::Tensor flash_decoding_attention(
    const torch::Tensor &Q,
    const torch::Tensor &K,
    const torch::Tensor &V
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add, "Vector Addition (CUDA)");
    m.def("matmul", &matrix_multiplication, "Matrix Multiplication (CUDA)");
    m.def("conv1D", &convolution_1D, "Convolution 1D (CUDA)");
    m.def("reverse_array", &reverse_array, "Reverse Array (CUDA)");
    m.def("mat_transpose", &matrix_transpose, "Matrix Transpose (CUDA)");
    m.def("color_invert", &color_invert, "Color Invert (CUDA)");
    m.def("relu", &relu, "ReLU (CUDA)");
    m.def("leaky_relu", &leaky_relu, "Leaky ReLU (CUDA)");
    m.def("rainbow_table", &rainbow_table, "Rainbow Table (CUDA)");
    m.def("matrix_copy", &matrix_copy, "Matrix Copy (CUDA)");
    m.def("count_equal", &count_equal, "Count Equal (CUDA)");
    m.def("count_equal_2D", &count_equal_2D, "Count Equal 2D (CUDA)");
    m.def("silu", &silu, "Silu (CUDA)");
    m.def("swiglu", &swiglu, "Swiglu (CUDA)");
    m.def("reduce_sum", &reduce_sum, "Reduce Sum (CUDA)");
    m.def("softmax_1D", &softmax_1D, "Softmax 1D (CUDA)");
    m.def("softmax_attention", &softmax_attention, "Softmax Attention (CUDA)");
    m.def("conv2D", &convolution_2D, "Convolution 2D (CUDA)");
    m.def("histogram", &histogram, "Histogram (CUDA)");
    m.def("prefix_sum", &prefix_sum, "Prefix Sum (CUDA)");
    m.def("dot_product", &dot_product, "Dot Product (CUDA)");
    m.def("sparse_matrix_vector_multiplication", &sparse_matrix_vector_multiplication, "Sparse Matrix Vector Multiplication (CUDA)");
    m.def("gemm", &gemm, "GEMM (CUDA)");
    m.def("avg_categorical_cross_entropy_loss", &avg_categorical_cross_entropy_loss, "Average Categorical Cross Entropy Loss (CUDA)");
    m.def("mse_vec", &mse_vec, "Mean Squared Error (CUDA)");
    m.def("gaussian_blur", &gaussian_blur, "Gaussian Blur (CUDA)");
    m.def("topk_1D", &topk_1D, "TopK (CUDA)");
    m.def("batched_matmul", &batched_matmul, "Batched Matrix Multiplication (CUDA)");
    m.def("int8_quantized_matmul", &int8_quantized_matmul, "Int8 Quantized Matrix Multiplication (CUDA)");
    m.def("ordinary_least_squares", &ordinary_least_squares, "Ordinary Least Squares (CUDA)");
    m.def("logistic_regression", &logistic_regression, "Logistic Regression (CUDA)");
    m.def("monte_carlo_integration", &monte_carlo_integration, "Monte Carlo Integration (CUDA)");
    m.def("sorting", &sorting, "Sorting (CUDA)");
    m.def("radix_sort", &radix_sort, "Radix Sort (CUDA)");
    m.def("matrix_power", &matrix_power, "Matrix Power (CUDA)");
    m.def("nearest_neighbor", &nearest_neighbor, "Nearest Neighbor (CUDA)");
    m.def("batch_norm", &batch_norm, "Batch Normalization (CUDA)");
    m.def("max_pooling_2D", &max_pooling_2D, "Max Pooling 2D (CUDA)");
    m.def("count_equal_3D", &count_equal_3D, "Count Equal 3D (CUDA)");
    m.def("subarray_sum", &subarray_sum, "Subarray Sum (CUDA)");
    m.def("subarray_sum_2D", &subarray_sum_2D, "Subarray Sum 2D (CUDA)");
    m.def("subarray_sum_3D", &subarray_sum_3D, "Subarray Sum 3D (CUDA)");
    m.def("rms_norm_1D", &rms_norm_1D, "RMS Norm (CUDA)");
    m.def("max_subarray_sum", &max_subarray_sum, "Max Subarray Sum (CUDA)");
    m.def("attention_with_linear_bias", &attention_with_linear_bias, "Attention with Linear Bias (CUDA)");
    m.def("batched_matrix_mul_fp16", &batched_matrix_mul_fp16, "Batched Matrix Multiplication (FP16) (CUDA)");
    m.def("dot_product_fp16", &dot_product_fp16, "Dot Product (FP16) (CUDA)");
    m.def("rope_embedding", &rope_embedding, "Rope Embedding (CUDA)");
    m.def("conv3D", &conv3D, "Convolution 3D (CUDA)");
    m.def("mha", &mha, "Multi-Head Attention (CUDA)");
    m.def("multi_agent_sim", &multi_agent_sim, "Multi-Agent Simulation (CUDA)");
    m.def("kmeans_clustering", &kmeans_clustering, "K-Means Clustering (CUDA)");
    m.def("causal_self_attention", &causal_self_attention, "Causal Self Attention (CUDA)");
    m.def("linear_self_attention", &linear_self_attention, "Linear Self Attention (CUDA)");
    m.def("sliding_window_self_attention", &sliding_window_self_attention, "Sliding Window Self Attention (CUDA)");
    m.def("weight_dequantization", &weight_dequantization, "Weight Dequantization (CUDA)");
    m.def("jacobi_stencil_2D", &jacobi_stencil_2D, "Jacobi Stencil 2D (CUDA)");
    m.def("stream_compaction", &stream_compaction, "Stream Compaction (CUDA)");
    m.def("moe_topk_gating", &moe_topk_gating, "MoE TopK Gating (CUDA)");
    m.def("gpt2_transformer_block", &gpt2_transformer_block, "GPT-2 Transformer Block (CUDA)");
    m.def("all_pairs_shortest_paths", &all_pairs_shortest_paths, "All Pairs Shortest Paths (CUDA)");
    m.def("gqa", &gqa, "GQA (CUDA)");
    m.def("linear_recurrence", &linear_recurrence, "Linear Recurrence (CUDA)");
    m.def("lora_linear", &lora_linear, "LoRA Linear (CUDA)");
    m.def("W4A16_matmul", &int4_weight_only_quantized_matMul, "W4A16_matmul (CUDA)");
    m.def("topp_sampling", &topp_sampling, "Top-P Sampling (CUDA)");
    m.def("fft_2D", &fft_2D, "FFT 2D (CUDA)");
    m.def("fast_fft", &fast_fft, "Fast FFT (CUDA)");
    m.def("int8_kvcache_attntion", &int8_kvcache_attntion, "Int8 KV Cache Attention (CUDA)");
    m.def("swiglu_mlp_block", &swiglu_mlp_block, "SwiGLU MLP Block (CUDA)");
    m.def("causal_depthwise_conv1D", &causal_depthwise_conv1D, "Causal Depthwise Conv1d (CUDA)");
    m.def("llama_transformer_block", &llama_transformer_block, "Llama Transformer Block (CUDA)");
    m.def("ssm_selective_scan", &ssm_selective_scan, "SSM Selective Scan (CUDA)");
    m.def("group_normalization", &group_normalization, "Group Normalization (CUDA)");
    m.def("token_embedding_layer", &token_embedding_layer, "Token Embedding Layer (CUDA)");
    m.def("attention_with_sinks", &attention_with_sinks, "Attention with Sinks (CUDA)");
    m.def("grpo_surrogate_loss", &grpo_surrogate_loss, "GRPO Surrogate Loss (CUDA)");
    m.def("ppo_clipped_surrogate_loss", &ppo_clipped_surrogate_loss, "PPO Clipped Surrogate Loss (CUDA)");
    m.def("dpo_sequence_loss", &dpo_sequence_loss, "DPO Sequence Loss (CUDA)");
    m.def("softmax_attention_backward", &softmax_attention_backward, "Softmax Attention Backward (CUDA)");
    m.def("adder_transformer_inference", &adder_transformer_inference, "Adder Transformer Inference (CUDA)");
    m.def("multi_head_cross_attention", &multi_head_cross_attention, "Multi-Head Cross Attention (CUDA)");
    m.def("layer_normalization", &layer_normalization, "Layer Normalization (CUDA)");
    m.def("gelu", &gelu, "GELU (CUDA)");
    m.def("distillation_kl_divergence", &distillation_kl_divergence, "Distillation KL Divergence (CUDA)");
    m.def("integral_image", &integral_image, "Integral Image (CUDA)");
    m.def("grouped_gemm", &grouped_gemm, "Grouped GEMM (CUDA)");
    m.def("rms_norm_backward", &rms_norm_backward, "RMS Norm Backward (CUDA)");
    m.def("layer_norm_backward", &layer_norm_backward, "Layer Norm Backward (CUDA)");
    m.def("multi_lora_batched_linear", &multi_lora_batched_linear, "Multi LoRA Batched Linear (CUDA)");
    m.def("fused_adamw", &fused_adamw, "Fused AdamW Optimizer (CUDA)");
    m.def("fused_ln_linear_gelu", &fused_ln_linear_gelu, "Fused LayerNorm + Linear + GELU (CUDA)");
    m.def("flash_decoding_attention", &flash_decoding_attention, "Flash Decoding Attention (CUDA)");
}