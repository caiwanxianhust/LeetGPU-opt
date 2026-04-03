#include <torch/extension.h>
#include <tuple>



// -------------------------- CUDA 函数声明 --------------------------
void vector_add_cuda(
    const float* A, const float* B, float* C, int N
);

void matrix_multiplication_cuda(
    const float* A, const float* B, float* C, int M, int N, int K
);

void matrix_multiplication_row_col_cuda(
    const float* A, const float* B, float* C, int M, int N, int K
);

void convolution_1d_cuda(
    const float* input, const float* kernel, float* output, int input_size, int kernel_size
);

void reverse_array_cuda(
    float* input, int N
);

void matrix_transpose_cuda(
    const float* input, float* output, int rows, int cols
);

void color_invert_cuda(
    unsigned char* image, int width, int height
);

void relu_cuda(
    const float* input, float* output, int N
);

void leaky_relu_cuda(
    const float* input, float* output, int N
);

void rainbow_table_cuda(
    const int* input, unsigned int* output, int N, int R
);

void copy_matrix_cuda(
    const float* A, float* B, int N
);

void count_equal_cuda(
    const int* input, int* output, int N, int K
);

void count_2d_equal_cuda(
    const int* input, int* output, int N, int M, int K
); 

void silu_cuda(
    const float* input, float* output, int N
);

void swiglu_cuda(
    const float* input, float* output, int N
);

void reduce_sum_cuda(
    const float* input, float* output, int N
);

void softmax_1D_cuda(
    const float* input, float* output, int N
);

void softmax_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int N, int d
);

void convolution_2d_cuda(
    const float* input, const float* kernel, float*  output, 
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
);

void histogram_cuda(
    const int* input, int* histogram, int N, int num_bins
);

void prefix_sum_cuda(
    float* input, float* output, float* reduce_sum_output, int N
);

void dot_product_cuda(
    float* A, float* B, float* result, int N
);

void matrix_vector_multiplication_cuda(
    const float* A, const float* x, float* y, int M, int N, int nnz
);

void gemm_cuda(
    const half* A, const half* B, half* C, int M, int N, int K, float alpha, float beta
); 

void avg_categorical_cross_entropy_loss_cuda(
    const float* logits, const int* true_labels, float* loss, int N, int C
);

void mean_squared_error_cuda(
    const float* predictions, const float* targets, float* mse, int N
);

void gaussian_blur_cuda(
    const float* input, const float* kernel, float* output,
    int input_rows, int input_cols, int kernel_rows, int kernel_cols
);

void topk_cuda(
    const float* input, float* output, int N, int k
);

void batched_matmul_cuda(
    const float* A, const float* B, float* C, int BATCH, int M, int N, int K
);

void int8_quantized_matmul_cuda(
    const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, int zero_point_A, int zero_point_B, int zero_point_C
);

void ordinary_least_squares_cuda(
    const float* X, const float* y, float* beta, int n_samples, int n_features
); 

void logistic_regression_cuda(
    const float* X, const float* y, float* beta, int n_samples, int n_features
);

void monte_carlo_integration_cuda(
    const float* y_samples, float* result, float a, float b, int n_samples
);

void sort_cuda(
    float* data, int N
);

void radix_sort_cuda(
    const unsigned int* input, unsigned int* output, int N
);

void matrix_power_cuda(
    const float* input, float* output, int N, int P
);

void nearest_neighbor_cuda(
    const float* points, int* indices, int N
);

void batch_norm_cuda(
    const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps
);

void max_pooling_2D_cuda(
    const float* input, float* output, int N, int C, int H, int W, int kernel_size, int stride, int padding
);

void count_array_element_3D_cuda(
    const int* input, int* output, int N, int M, int K, int P
);


void subarray_sum_cuda(
    const int* input, int* output, int N, int S, int E
);

void subarray_sum_2D_cuda(
    const int* input, int* output, int N, int M, int S_ROW, int E_ROW, int S_COL, int E_COL
);

void subarray_sum_3D_cuda(
    const int* input, int* output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL
);

void rms_norm_cuda(
    const float* input, float gamma, float beta, float* output, int N, float eps
);

void max_subarray_sum_cuda(
    const int* input, int* output, int N, int window_size
); 

void attention_with_linear_bias_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int N, int d, float alpha
);

void batched_matrix_mul_fp16_cuda(
    const half* A, const half* B, half* C, int BATCH, int M, int N, int K
);

void dot_product_fp16_cuda(
    const half* A, const half* B, half* result, int N
);

void rope_embedding_cuda(
    float* Q, float* cos, float* sin, float* output, int M, int D
);

void conv3D_cuda(
    const float* input, const float* kernel, float* output, int input_depth,
    int input_rows, int input_cols, int kernel_depth, int kernel_rows, int kernel_cols
); 

void mha_cuda(
    const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h
); 

void multi_agent_sim_cuda(
    const float* agents, float* agents_next, int N
); 

void kmeans_clustering_cuda(
    const float* data_x, const float* data_y, int* labels,
    float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
    float* final_centroid_y, int sample_size, int k, int max_iterations
);

void causal_self_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d
);

void linear_self_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d
);

void sliding_window_self_attention_cuda(
    const float* Q, const float* K, const float* V, float* output, int M, int d, int window_size
);

void weight_dequantization_cuda(
    const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE
);


void jacobi_stencil_2D_cuda(
    const float* input, float* output, int rows, int cols
);

void stream_compaction_cuda(
    const float* A, int N, float* out
); 

void moe_topk_gating_cuda(
    const float* logits, float* topk_weights, int* topk_indices, int M, int E, int k
);

void gpt2_transformer_block_cuda(
    const float* x, float* output, const float* weights, int seq_len
);


void all_pairs_shortest_paths_cuda(
    const float* dist, float* output, int N
); 

namespace GQA {
void gqa_cuda(
    const float* Q, const float* K, const float* V, float* output,
    int num_q_heads, int num_kv_heads, int seq_len, int head_dim
);
}

void linear_recurrence_cuda(
    const float* a, const float* x, float* h, int B, int L
);

void lora_linear_cuda(
    const float* x, const float* W, const float* A, const float* B, float* output,
    int batch, int d_in, int d_out, int rank, float lora_scale
); 




// -------------------------- Python 绑定函数 --------------------------
torch::Tensor vector_add(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have the same size");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");
    int N = A.size(0);
    auto options = A.options();
    torch::Tensor C = torch::empty({N}, options);
    vector_add_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    return C;
}


torch::Tensor matrix_multiplication(
    const torch::Tensor& A,
    const torch::Tensor& B, 
    const bool transpose_B = false
) { 
    TORCH_CHECK(A.ndimension() == 2, "Input A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "Input B must be 2D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");

    auto options = A.options();
    torch::Tensor C;

    if (transpose_B) {
        TORCH_CHECK(A.size(1) == B.size(1), "Input tensors must have compatible shapes");
        int M = A.size(0), N = B.size(0), K = A.size(1);
        C = torch::empty({M, N}, options);
        matrix_multiplication_row_col_cuda(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
        return C;
    }
    else {
        TORCH_CHECK(A.size(1) == B.size(0), "Input tensors must have compatible shapes");
        int M = A.size(0), N = B.size(1), K = A.size(1);
        C = torch::empty({M, N}, options);

        matrix_multiplication_cuda(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
        return C;
    }
}


torch::Tensor convolution_1D(
    const torch::Tensor& input,
    const torch::Tensor& kernel
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(kernel.ndimension() == 1, "Kernel must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(kernel.device().is_cuda(), "Kernel must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be of type float32");
    TORCH_CHECK(input.size(0) >= kernel.size(0), "Input must be longer than kernel");
    int input_size = input.size(0), kernel_size = kernel.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({input_size - kernel_size + 1}, options);
    convolution_1d_cuda(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        input_size, kernel_size
    );
    return output;
}


void reverse_array(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    reverse_array_cuda(
        input.data_ptr<float>(),
        N
    );
    return;
}


torch::Tensor matrix_transpose(
    const torch::Tensor& input
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int rows = input.size(0), cols = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({cols, rows}, options);
    matrix_transpose_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );
    return output;
}


void color_invert(
    const torch::Tensor& image, int width, int height
) {
    TORCH_CHECK(image.ndimension() == 1, "Image must be 1D");
    TORCH_CHECK(image.device().is_cuda(), "Image must be on CUDA");
    TORCH_CHECK(image.dtype() == torch::kUInt8, "Image must be of type uint8");
    color_invert_cuda(
        image.data_ptr<unsigned char>(),
        width, height
    );
    return;
}


torch::Tensor relu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    relu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor leaky_relu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    leaky_relu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor rainbow_table(
    const torch::Tensor& input,
    int R
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    int N = input.size(0);
    auto options = input.options().dtype(torch::kUInt32);  // 正确的写法
    torch::Tensor output = torch::empty({N}, options);
    rainbow_table_cuda(
        input.data_ptr<int>(),
        output.data_ptr<unsigned int>(),
        N, R
    );
    return output;
}


torch::Tensor matrix_copy(
    const torch::Tensor& A,
    int N
) {
    TORCH_CHECK(A.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(A.size(0) == N * N, "Input must be square matrix");
    auto options = A.options();
    torch::Tensor B = torch::empty({N * N}, options);
    copy_matrix_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        N
    );
    return B;
}


torch::Tensor count_equal(
    const torch::Tensor& input,
    int K
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    int N = input.size(0);
    torch::Tensor output = torch::zeros({1}, input.options());
    count_equal_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        N,
        K
    );
    return output;
}


torch::Tensor count_equal_2D(
    const torch::Tensor& input,
    int M, int N, int K
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) == M * N, "Input must be a 2D matrix");
    torch::Tensor output = torch::zeros({1}, input.options());
    count_2d_equal_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        M, N, K
    );
    return output;
}


torch::Tensor silu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    silu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor swiglu(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(input.size(0) % 2 == 0, "Input size must be even");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N / 2}, options);
    swiglu_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor reduce_sum(
    const torch::Tensor& input
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    torch::Tensor output = torch::zeros({1}, input.options());
    reduce_sum_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0)
    );
    return output;
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


torch::Tensor softmax_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.ndimension() == 2, "Input Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "Input K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "Input V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Input Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "Input K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "Input V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Input Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Input K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "Input V must be of type float32");
    TORCH_CHECK(Q.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(V.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(K.size(0) == V.size(0), "Input tensors must have compatible shapes");
    int M = Q.size(0), N = K.size(0), d = Q.size(1);
    auto options = Q.options();
    torch::Tensor output = torch::empty({M, d}, options);
    softmax_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, d
    );
    return output;
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

torch::Tensor histogram(
    const torch::Tensor& input,
    int num_bins
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor histogram = torch::zeros({num_bins}, options);
    histogram_cuda(
        input.data_ptr<int>(),
        histogram.data_ptr<int>(),
        N,
        num_bins
    );
    return histogram;
}


torch::Tensor prefix_sum(
    const torch::Tensor& input
) { 
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    torch::Tensor output = torch::empty({N}, input.options());
    torch::Tensor reduce_sum_output = torch::empty({N}, input.options());
    prefix_sum_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        reduce_sum_output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor dot_product(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have the same size");
    
    int N = A.size(0);
    auto options = A.options();
    torch::Tensor result = torch::zeros({1}, options);
    
    dot_product_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        result.data_ptr<float>(),
        N
    );
    
    return result;
}


torch::Tensor sparse_matrix_vector_multiplication(
    const torch::Tensor& A, 
    const torch::Tensor& x, 
    int M, int N, int nnz
) {
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(x.ndimension() == 1, "Input x must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(x.device().is_cuda(), "Input x must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be of type float32");
    TORCH_CHECK(A.size(0) == M * N, "Input A must be a 1D matrix")
    TORCH_CHECK(x.size(0) == N, "Input x must be a 1D matrix")
    torch::Tensor y = torch::empty({M}, x.options());
    matrix_vector_multiplication_cuda(
        A.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N, nnz
    );
    return y;
}


torch::Tensor gemm(
    const torch::Tensor& A,
    const torch::Tensor& B,
    torch::Tensor& C,
    int M, int N, int K,
    float alpha, float beta
) { 
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(C.ndimension() == 1, "Input C must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(C.device().is_cuda(), "Input C must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be of type float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be of type float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "Input C must be of type float16");
    TORCH_CHECK(A.size(0) == M * K, "Input A must be a 1D matrix")
    TORCH_CHECK(B.size(0) == N * K, "Input B must be a 1D matrix")
    TORCH_CHECK(C.size(0) == M * N, "Input C must be a 1D matrix")
    
    gemm_cuda(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K,
        alpha, beta
    );
    return C;
}
    

torch::Tensor avg_categorical_cross_entropy_loss(
    const torch::Tensor& logits,
    const torch::Tensor& true_labels,
    int N, int C
) {
    TORCH_CHECK(logits.ndimension() == 1, "Input logits must be 1D");
    TORCH_CHECK(true_labels.ndimension() == 1, "Input true_labels must be 1D");
    TORCH_CHECK(logits.device().is_cuda(), "Input logits must be on CUDA");
    TORCH_CHECK(true_labels.device().is_cuda(), "Input true_labels must be on CUDA");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Input logits must be of type float32");
    TORCH_CHECK(true_labels.dtype() == torch::kInt32, "Input true_labels must be of type int32");
    TORCH_CHECK(logits.size(0) == N * C, "Input logits must be a 1D matrix")
    TORCH_CHECK(true_labels.size(0) == N, "Input true_labels must be a 1D matrix")
    torch::Tensor loss = torch::empty({1}, logits.options());
    avg_categorical_cross_entropy_loss_cuda(
        logits.data_ptr<float>(),
        true_labels.data_ptr<int>(),
        loss.data_ptr<float>(),
        N, C
    );
    return loss;
}


torch::Tensor mse_vec(
    const torch::Tensor& predictions,
    const torch::Tensor& targets
) {
    TORCH_CHECK(predictions.ndimension() == 1, "Input predictions must be 1D");
    TORCH_CHECK(targets.ndimension() == 1, "Input targets must be 1D");
    TORCH_CHECK(predictions.device().is_cuda(), "Input predictions must be on CUDA");
    TORCH_CHECK(targets.device().is_cuda(), "Input targets must be on CUDA");
    TORCH_CHECK(predictions.dtype() == torch::kFloat32, "Input predictions must be of type float32");
    TORCH_CHECK(targets.dtype() == torch::kFloat32, "Input targets must be of type float32");
    TORCH_CHECK(predictions.size(0) == targets.size(0), "Input tensors must have the same size");
    
    int N = predictions.size(0);
    torch::Tensor mse = torch::empty({1}, predictions.options());
    
    mean_squared_error_cuda(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        mse.data_ptr<float>(),
        N
    );
    
    return mse;
}


torch::Tensor gaussian_blur(
    const torch::Tensor& input,
    const torch::Tensor& kernel
) {
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D (H, W)");
    TORCH_CHECK(kernel.ndimension() == 2, "Kernel must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(kernel.device().is_cuda(), "Kernel must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(kernel.dtype() == torch::kFloat32, "Kernel must be of type float32");

    torch::Tensor output = torch::empty_like(input);
    int input_rows = input.size(0);
    int input_cols = input.size(1);
    int kernel_rows = kernel.size(0);
    int kernel_cols = kernel.size(1);
    TORCH_CHECK(kernel_rows % 2 == 1 && kernel_cols % 2 == 1, "Kernel size must be odd");
    TORCH_CHECK(input_rows %2 == 1 && input_cols % 2 == 1, "Input size must be odd");

    gaussian_blur_cuda(
        input.data_ptr<float>(),
        kernel.data_ptr<float>(),
        output.data_ptr<float>(),
        input_rows, input_cols,
        kernel_rows, kernel_cols
    );
    return output;
}


torch::Tensor topk_1D(
    const torch::Tensor& input,
    int k
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(k > 0 && k <= input.size(0), "k must be in the range [1, N]");

    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({k}, options);
    
    topk_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, k
    );
    
    return output; 
}


torch::Tensor batched_matmul(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 3, "Input A must be 3D");
    TORCH_CHECK(B.ndimension() == 3, "Input B must be 3D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have the same batch size");
    TORCH_CHECK(A.size(2) == B.size(1), "Input tensors must have compatible shapes");

    int BATCH = A.size(0);
    int M = A.size(1);
    int N = B.size(2);
    int K = A.size(2);

    auto options = A.options();
    torch::Tensor C = torch::empty({BATCH, M, N}, options);

    batched_matmul_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, M, N, K
    );

    return C;
}


torch::Tensor int8_quantized_matmul(
    const torch::Tensor& A,
    const torch::Tensor& B,
    float scale_A, float scale_B, float scale_C,
    int zero_point_A, int zero_point_B, int zero_point_C
) {
    TORCH_CHECK(A.ndimension() == 2, "Input A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "Input B must be 2D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kInt8, "Input A must be of type int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "Input B must be of type int8");
    TORCH_CHECK(A.size(1) == B.size(0), "Input tensors must have compatible shapes");

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    auto options = A.options();
    torch::Tensor C = torch::empty({M, N}, options.dtype(torch::kInt8));

    int8_quantized_matmul_cuda(
        A.data_ptr<int8_t>(),
        B.data_ptr<int8_t>(),
        C.data_ptr<int8_t>(),
        M, N, K,
        scale_A, scale_B, scale_C,
        zero_point_A, zero_point_B, zero_point_C
    );
    return C;
}


torch::Tensor ordinary_least_squares(
    const torch::Tensor& X,
    const torch::Tensor& y
) { 
    TORCH_CHECK(X.ndimension() == 2, "Input X must be 2D");
    TORCH_CHECK(y.ndimension() == 1, "Input y must be 1D");
    TORCH_CHECK(X.device().is_cuda(), "Input X must be on CUDA");
    TORCH_CHECK(y.device().is_cuda(), "Input y must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "Input X must be of type float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "Input y must be of type float32");
    TORCH_CHECK(X.size(0) == y.size(0), "Input tensors must have the same size");

    torch::Tensor beta = torch::empty({X.size(1)}, X.options());
    ordinary_least_squares_cuda(
        X.data_ptr<float>(),
        y.data_ptr<float>(),
        beta.data_ptr<float>(),
        X.size(0), X.size(1)
    );

    return beta;
}


torch::Tensor logistic_regression(
    const torch::Tensor& X,
    const torch::Tensor& y,
    const torch::Tensor& beta
) { 
    TORCH_CHECK(X.ndimension() == 2, "Input X must be 2D");
    TORCH_CHECK(y.ndimension() == 1, "Input y must be 1D");
    TORCH_CHECK(beta.ndimension() == 1, "Input beta must be 1D");
    TORCH_CHECK(X.device().is_cuda(), "Input X must be on CUDA");
    TORCH_CHECK(y.device().is_cuda(), "Input y must be on CUDA");
    TORCH_CHECK(beta.device().is_cuda(), "Input beta must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "Input X must be of type float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "Input y must be of type float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "Input beta must be of type float32");
    TORCH_CHECK(X.size(0) == y.size(0), "Input tensors must have the same size");
    TORCH_CHECK(X.size(1) == beta.size(0), "Input tensors must have compatible shapes");
    logistic_regression_cuda(
        X.data_ptr<float>(),
        y.data_ptr<float>(),
        beta.data_ptr<float>(),
        X.size(0), X.size(1)
    );
    return beta;
}

torch::Tensor monte_carlo_integration(
    const torch::Tensor& y_samples,
    float a, float b
) {
    TORCH_CHECK(y_samples.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(y_samples.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(y_samples.dtype() == torch::kFloat32, "Input must be of type float32");

    int N = y_samples.size(0);
    torch::Tensor result = torch::empty({1}, y_samples.options());
    monte_carlo_integration_cuda(
        y_samples.data_ptr<float>(),
        result.data_ptr<float>(),
        a, b, N
    );
    return result;
}


void sorting(const torch::Tensor& input) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    sort_cuda(input.data_ptr<float>(), input.size(0));
    return;
}


torch::Tensor radix_sort(
    const torch::Tensor& input
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kUInt32, "Input must be of type uint32");

    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);

    radix_sort_cuda(
        input.data_ptr<unsigned int>(),
        output.data_ptr<unsigned int>(),
        N
    );
    return output;
}


torch::Tensor matrix_power(
    const torch::Tensor& input,
    int power
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(power >= 0, "Power must be non-negative");
    TORCH_CHECK(input.size(0) == input.size(1), "Input must be square");

    torch::Tensor output = torch::empty_like(input);

    matrix_power_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        power
    );

    return output;
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


torch::Tensor max_pooling_2D(
    const torch::Tensor& input,
    int kernel_size,
    int stride,
    int padding
) { 
    TORCH_CHECK(input.ndimension() == 4, "Input must be 4D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32")

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    torch::Tensor output = torch::empty({N, C, H_out, W_out}, input.options());

    max_pooling_2D_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_size, stride, padding
    );
    return output;
}


torch::Tensor batch_norm(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    float eps
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(gamma.ndimension() == 1, "Gamma must be 1D");
    TORCH_CHECK(beta.ndimension() == 1, "Beta must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(gamma.device().is_cuda(), "Gamma must be on CUDA");
    TORCH_CHECK(beta.device().is_cuda(), "Beta must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(gamma.dtype() == torch::kFloat32, "Gamma must be of type float32");
    TORCH_CHECK(beta.dtype() == torch::kFloat32, "Beta must be of type float32");
    TORCH_CHECK(input.size(1) == gamma.size(0), "Input and gamma must have compatible shapes");
    TORCH_CHECK(input.size(1) == beta.size(0), "Input and beta must have compatible shapes");

    int N = input.size(0);
    int D = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({N, D}, options);

    batch_norm_cuda(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D, eps
    );

    return output;
}


torch::Tensor count_equal_3D(
    const torch::Tensor& input,
    const int n, const int m, const int k,
    const int p
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) == n * m * k, "Input must be a 3D matrix");

    torch::Tensor output = torch::zeros({1}, input.options());

    count_array_element_3D_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, m, k, p
    );

    return output;
}


torch::Tensor subarray_sum(
    const torch::Tensor& input,
    const int S, const int E
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(input.size(0) >= S && input.size(0) >= E, "Invalid range");
    TORCH_CHECK(S < E, "Invalid range");

    int n = input.size(0);
    torch::Tensor output = torch::zeros({1}, input.options());
    subarray_sum_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, S, E
    );
    return output;
}


torch::Tensor subarray_sum_2D(
    const torch::Tensor& input,
    const int S_ROW, const int E_ROW, const int S_COL, const int E_COL
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(E_ROW >= S_ROW && input.size(0) >= E_ROW, "Invalid range");
    TORCH_CHECK(E_COL >= S_COL && input.size(1) >= E_COL, "Invalid range");

    int n = input.size(0);
    int m = input.size(1);
    
    torch::Tensor output = torch::zeros({1}, input.options());
    subarray_sum_2D_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, m,
        S_ROW, E_ROW, S_COL, E_COL
    );
    return output;
}


torch::Tensor subarray_sum_3D(
    const torch::Tensor& input,
    const int S_DEP, const int E_DEP, const int S_ROW, const int E_ROW, const int S_COL, const int E_COL
) {
    TORCH_CHECK(input.ndimension() == 3, "Input must be 3D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(E_DEP >= S_DEP && input.size(0) >= E_DEP, "Invalid range");
    TORCH_CHECK(E_ROW >= S_ROW && input.size(1) >= E_ROW, "Invalid range");
    TORCH_CHECK(E_COL >= S_COL && input.size(2) >= E_COL, "Invalid range");

    int n = input.size(0);
    int m = input.size(1);
    int k = input.size(2);
    torch::Tensor output = torch::empty({1}, input.options());
    subarray_sum_3D_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        n, m, k,
        S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL
    );
    return output;
}


torch::Tensor rms_norm_1D(
    const torch::Tensor& input,
    float eps, float gamma, float beta
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({N}, options);
    rms_norm_cuda(
        input.data_ptr<float>(),
        gamma,
        beta,
        output.data_ptr<float>(),
        N,
        eps
    );
    return output;
}


torch::Tensor max_subarray_sum(
    const torch::Tensor& input,
    const int window_size
) {
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dtype() == torch::kInt32, "Input must be of type int32");
    TORCH_CHECK(window_size > 0 && window_size <= input.size(0), "Invalid window size");

    int N = input.size(0);
    auto options = input.options();
    torch::Tensor output = torch::empty({1}, options);
    max_subarray_sum_cuda(
        input.data_ptr<int>(),
        output.data_ptr<int>(),
        N,
        window_size
    );
    return output;
}


torch::Tensor attention_with_linear_bias(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    float alpha
) { 
    TORCH_CHECK(Q.ndimension() == 2, "Input Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "Input K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "Input V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Input Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "Input K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "Input V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Input Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "Input K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "Input V must be of type float32");
    TORCH_CHECK(Q.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(V.size(1) == K.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(K.size(0) == V.size(0), "Input tensors must have compatible shapes");
    int M = Q.size(0), N = K.size(0), d = Q.size(1);
    auto options = Q.options();
    torch::Tensor output = torch::empty({M, d}, options);
    attention_with_linear_bias_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, d,
        alpha
    );
    return output;
}


torch::Tensor batched_matrix_mul_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 3, "Input A must be 3D");
    TORCH_CHECK(B.ndimension() == 3, "Input B must be 3D    ");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have compatible shapes");
    TORCH_CHECK(A.size(2) == B.size(1), "Input tensors must have compatible shapes");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be of type float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be of type float16");

    int BATCH = A.size(0), M = A.size(1), K = A.size(2), N = B.size(2);
    torch::Tensor C = torch::empty({BATCH, M, N}, A.options());
    batched_matrix_mul_fp16_cuda(
        reinterpret_cast<half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        BATCH, M, N, K
    );
    return C;
}


torch::Tensor dot_product_fp16(
    const torch::Tensor& A,
    const torch::Tensor& B
) {
    TORCH_CHECK(A.ndimension() == 1, "Input A must be 1D");
    TORCH_CHECK(B.ndimension() == 1, "Input B must be 1D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input A must be of type float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "Input B must be of type float16");
    TORCH_CHECK(A.size(0) == B.size(0), "Input tensors must have the same size");

    int N = A.size(0);
    auto options = A.options();
    torch::Tensor result = torch::zeros({1}, options.dtype(torch::kFloat16));

    dot_product_fp16_cuda(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(result.data_ptr<at::Half>()),
        N
    );

    return result;
}


torch::Tensor rope_embedding(
    const torch::Tensor& input,
    const torch::Tensor& cos_table,
    const torch::Tensor& sin_table
) { 
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    TORCH_CHECK(cos_table.ndimension() == 2, "cos_table must be 2D");
    TORCH_CHECK(sin_table.ndimension() == 2, "sin_table must be 2D");
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(cos_table.device().is_cuda(), "cos_table must be on CUDA");
    TORCH_CHECK(sin_table.device().is_cuda(), "sin_table must be on CUDA");
    TORCH_CHECK(input.size(0) == cos_table.size(0), "Input and cos_table must have the same number of rows");
    TORCH_CHECK(input.size(0) == sin_table.size(0), "Input and sin_table must have the same number of rows");
    TORCH_CHECK(input.size(1) == cos_table.size(1), "Input and cos_table must have the same number of columns");
    TORCH_CHECK(input.size(1) == sin_table.size(1), "Input and sin_table must have the same number of columns");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(cos_table.dtype() == torch::kFloat32, "cos_table must be of type float32");
    TORCH_CHECK(sin_table.dtype() == torch::kFloat32, "sin_table must be of type float32");
    TORCH_CHECK(input.size(1) % 2 == 0, "input must have an even number of columns");

    int M = input.size(0);
    int D = input.size(1);
    auto options = input.options();
    torch::Tensor output = torch::empty({M, D}, options);

    rope_embedding_cuda(
        input.data_ptr<float>(),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        output.data_ptr<float>(),
        M, D
    );
    
    return output;
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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kmeans_clustering(
    const torch::Tensor& data_x,
    const torch::Tensor& data_y,
    const torch::Tensor& init_centroids_x,
    const torch::Tensor& init_centroids_y,
    const int max_iterations
) { 
    TORCH_CHECK(data_x.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(data_y.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(init_centroids_x.ndimension() == 1, "init_centroids_x must be 1D");
    TORCH_CHECK(init_centroids_y.ndimension() == 1, "init_centroids_y must be 1D");
    TORCH_CHECK(data_x.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(data_y.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(init_centroids_y.device().is_cuda(), "init_centroids_y must be on CUDA");
    TORCH_CHECK(init_centroids_x.device().is_cuda(), "init_centroids_x must be on CUDA");
    TORCH_CHECK(data_x.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(data_y.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(init_centroids_x.dtype() == torch::kFloat32, "init_centroids_x must be of type float32");
    TORCH_CHECK(init_centroids_y.dtype() == torch::kFloat32, "init_centroids_y must be of type float32");

    int sample_size = data_x.size(0);
    int k = init_centroids_x.size(0);
    TORCH_CHECK(sample_size > k, "Sample size must be greater than number of clusters");
    TORCH_CHECK(data_y.size(0) == sample_size, "Input must have the same number of rows as init_centroids_y");
    TORCH_CHECK(init_centroids_y.size(0) == k, "init_centroids_y must have the same number of rows as init_centroids_x");

    torch::Tensor labels = torch::empty({sample_size}, data_x.options().dtype(torch::kInt32));
    torch::Tensor final_centroids_x = torch::empty({k}, data_x.options());
    torch::Tensor final_centroids_y = torch::empty({k}, data_x.options());
    kmeans_clustering_cuda(
        data_x.data_ptr<float>(), 
        data_y.data_ptr<float>(), 
        labels.data_ptr<int>(), 
        init_centroids_x.data_ptr<float>(),
        init_centroids_y.data_ptr<float>(),
        final_centroids_x.data_ptr<float>(),
        final_centroids_y.data_ptr<float>(),
        sample_size, k, max_iterations
    );
    return {labels, final_centroids_x, final_centroids_y};
}


torch::Tensor mha(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int heads
) { 
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");
    TORCH_CHECK(Q.size(1) % heads == 0, "Q size must be divisible by heads");

    int N = Q.size(0);
    int d_model = Q.size(1);
    
    torch::Tensor output = torch::zeros({N, d_model}, Q.options());
    torch::Tensor attn_scores = torch::zeros({heads, N, N}, Q.options());
    mha_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        N, d_model, heads
    );

    return output;
}


torch::Tensor multi_agent_sim(
    const torch::Tensor& agents
) {
    TORCH_CHECK(agents.ndimension() == 2, "Agents must be 2D");
    TORCH_CHECK(agents.device().is_cuda(), "Agents must be on CUDA");
    TORCH_CHECK(agents.dtype() == torch::kFloat32, "Agents must be of type float32");
    TORCH_CHECK(agents.size(1) == 4, "Each agent must have 4 attributes (x, y, vx, vy)");

    int N = agents.size(0);
    auto options = agents.options();
    torch::Tensor updated_agents = torch::empty({N, 4}, options);

    multi_agent_sim_cuda(
        agents.data_ptr<float>(),
        updated_agents.data_ptr<float>(),
        N
    );

    return updated_agents;
}


torch::Tensor causal_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");

    int M = Q.size(0);
    int d_model = Q.size(1);
    torch::Tensor output = torch::zeros({M, d_model}, Q.options());

    causal_self_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, d_model
    );
    return output;
}


torch::Tensor linear_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");

    int M = Q.size(0);
    int d_model = Q.size(1);
    torch::Tensor output = torch::zeros({M, d_model}, Q.options());

    linear_self_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, d_model
    );
    return output;
}


torch::Tensor sliding_window_self_attention(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const int window_size
) { 
    TORCH_CHECK(Q.ndimension() == 2, "Q must be 2D");
    TORCH_CHECK(K.ndimension() == 2, "K must be 2D");
    TORCH_CHECK(V.ndimension() == 2, "V must be 2D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    TORCH_CHECK(Q.size(0) == K.size(0), "Q and K must have the same number of rows");
    TORCH_CHECK(Q.size(1) == K.size(1), "Q and K must have the same number of columns");
    TORCH_CHECK(Q.size(0) == V.size(0), "Q and V must have the same number of rows");
    TORCH_CHECK(Q.size(1) == V.size(1), "Q and V must have the same number of columns");
    TORCH_CHECK(window_size > 0 && window_size <= 32, "Window size must be positive");

    int M = Q.size(0);
    int d_model = Q.size(1);
    torch::Tensor output = torch::zeros({M, d_model}, Q.options());

    sliding_window_self_attention_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        M, d_model, window_size
    );
    return output;
}


torch::Tensor weight_dequantization(
    const torch::Tensor& X,
    const torch::Tensor& S, 
    const int tile_size
) {
    TORCH_CHECK(X.ndimension() == 2, "X must be 2D");
    TORCH_CHECK(S.ndimension() == 2, "S must be 2D");
    TORCH_CHECK(X.device().is_cuda(), "X must be on CUDA");
    TORCH_CHECK(S.device().is_cuda(), "S must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be of type float32");
    TORCH_CHECK(S.dtype() == torch::kFloat32, "S must be of type float32");
    int M = X.size(0);
    int N = X.size(1);
    TORCH_CHECK(((M + tile_size - 1) / tile_size) == S.size(0), "Invalid shape for S");
    TORCH_CHECK(((N + tile_size - 1) / tile_size) == S.size(1), "Invalid shape for S");

    torch::Tensor output = torch::empty({M, N}, X.options());

    weight_dequantization_cuda(
        X.data_ptr<float>(),
        S.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, tile_size
    );
    return output;
}


torch::Tensor jacobi_stencil_2D(const torch::Tensor& input)
{
    int cols = input.size(1);
    int rows = input.size(0);
    TORCH_CHECK(input.ndimension() == 2, "Input must be 2D");
    torch::Tensor output = torch::empty_like(input);
    jacobi_stencil_2D_cuda(input.data_ptr<float>(), output.data_ptr<float>(), rows, cols);
    return output;
}

torch::Tensor stream_compaction(const torch::Tensor& input)
{
    int N = input.size(0);
    TORCH_CHECK(input.ndimension() == 1, "Input must be 1D");
    torch::Tensor output = torch::zeros_like(input);
    stream_compaction_cuda(input.data_ptr<float>(), N, output.data_ptr<float>());
    return output;
}


std::tuple<torch::Tensor, torch::Tensor> moe_topk_gating(const torch::Tensor& logits, int k)
{ 
    TORCH_CHECK(logits.ndimension() == 2, "Logits must be 2D");
    TORCH_CHECK(logits.device().is_cuda(), "Logits must be on CUDA");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Logits must be of type float32");
    TORCH_CHECK(k > 0 && k <= logits.size(1), "Invalid k");
    int M = logits.size(0);
    int E = logits.size(1);
    torch::Tensor topk_weights = torch::empty({M, k}, logits.options());
    torch::Tensor topk_indices = torch::empty({M, k}, logits.options().dtype(torch::kInt32));
    moe_topk_gating_cuda(
        logits.data_ptr<float>(), 
        topk_weights.data_ptr<float>(), 
        topk_indices.data_ptr<int>(), 
        M, E, k);
    return {topk_weights, topk_indices};
}


torch::Tensor gpt2_transformer_block(
    const torch::Tensor& x,
    const torch::Tensor& weights
) {
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(weights.ndimension() == 1, "weights must be 1D");

    int seq_len = x.size(0);
    int d_model = x.size(1);
    torch::Tensor output = torch::empty({seq_len, d_model}, x.options());
    gpt2_transformer_block_cuda(
        x.data_ptr<float>(), 
        output.data_ptr<float>(), 
        weights.data_ptr<float>(), 
        seq_len);
    return output;
}


torch::Tensor all_pairs_shortest_paths(
    const torch::Tensor& graph
) {
    TORCH_CHECK(graph.ndimension() == 2, "Graph must be 2D");
    TORCH_CHECK(graph.device().is_cuda(), "Graph must be on CUDA");
    TORCH_CHECK(graph.dtype() == torch::kFloat32, "Graph must be of type float32");
    TORCH_CHECK(graph.size(0) == graph.size(1), "Graph must be a square matrix");

    int N = graph.size(0);
    torch::Tensor output = torch::empty_like(graph);
    all_pairs_shortest_paths_cuda(
        graph.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}


torch::Tensor gqa(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.ndimension() == 3, "Q must be 3D");
    TORCH_CHECK(K.ndimension() == 3, "K must be 3D");
    TORCH_CHECK(V.ndimension() == 3, "V must be 3D");
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be of type float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be of type float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be of type float32");
    int num_q_heads = Q.size(0);
    int num_kv_heads = K.size(0);
    int seq_len = Q.size(1);
    int head_dim = Q.size(2);

    TORCH_CHECK(K.size(1) == seq_len, "K must have the same sequence length as Q");
    TORCH_CHECK(V.size(1) == seq_len, "V must have the same sequence length as Q");
    TORCH_CHECK(K.size(2) == head_dim, "K must have the same head dimension as Q");
    TORCH_CHECK(V.size(2) == head_dim, "V must have the same head dimension as Q");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "Number of Q heads must be a multiple of number of KV heads");

    torch::Tensor output = torch::empty({num_q_heads, seq_len, head_dim}, Q.options());
    GQA::gqa_cuda(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        num_q_heads, num_kv_heads, seq_len, head_dim
    );
    return output;
}

torch::Tensor linear_recurrence(
    const torch::Tensor& a,
    const torch::Tensor& x
) {
    TORCH_CHECK(a.ndimension() == 2, "a must be 2D");
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(a.size(0) == x.size(0), "The number of rows in a must match the number of rows in x");
    TORCH_CHECK(a.size(1) == x.size(1), "The number of columns in a must match the number of columns in x");
    TORCH_CHECK(a.device().is_cuda(), "a must be on CUDA");
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be of type float32");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be of type float32");

    int B = a.size(0);
    int L = a.size(1);

    torch::Tensor output = torch::empty_like(x);
    linear_recurrence_cuda(
        a.data_ptr<float>(),
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, L
    );
    return output;
}


torch::Tensor lora_linear(
    const torch::Tensor& x,
    const torch::Tensor& W,
    const torch::Tensor& A,
    const torch::Tensor& B,
    float lora_scale
) { 
    int batch = x.size(0);
    int d_in = x.size(1);
    int d_out = W.size(0);
    int rank = A.size(0);
    TORCH_CHECK(x.ndimension() == 2, "x must be 2D");
    TORCH_CHECK(W.ndimension() == 2, "W must be 2D");
    TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "B must be 2D");
    TORCH_CHECK(x.device().is_cuda(), "x must be on CUDA");
    TORCH_CHECK(W.device().is_cuda(), "W must be on CUDA"); 
    TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be of type float32");
    TORCH_CHECK(W.dtype() == torch::kFloat32, "W must be of type float32");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be of type float32");
    TORCH_CHECK(x.size(1) == W.size(1), "The number of columns in x must match the number of columns in W");
    TORCH_CHECK(x.size(1) == A.size(1), "The number of columns in x must match the number of columns in A");
    TORCH_CHECK(W.size(0) == B.size(0), "The number of rows in W must match the number of rows in B");
    TORCH_CHECK(A.size(0) == B.size(1), "The number of rows in A must match the number of columns in B");

    torch::Tensor output = torch::empty({batch, d_out}, x.options());
    lora_linear_cuda(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, d_in, d_out, rank, lora_scale
    );
    return output;
}







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
}