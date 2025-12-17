#include <torch/extension.h>
#include <tuple>



// -------------------------- CUDA 函数声明 --------------------------
void vector_add_cuda(
    const float* A, const float* B, float* C, int N
);

void matrix_multiplication_cuda(
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
    const torch::Tensor& B
) { 
    TORCH_CHECK(A.ndimension() == 2, "Input A must be 2D");
    TORCH_CHECK(B.ndimension() == 2, "Input B must be 2D");
    TORCH_CHECK(A.device().is_cuda(), "Input A must be on CUDA");
    TORCH_CHECK(B.device().is_cuda(), "Input B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be of type float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be of type float32");
    TORCH_CHECK(A.size(1) == B.size(0), "Input tensors must have compatible shapes");
    int M = A.size(0), N = A.size(1), K = B.size(1);
    auto options = A.options();
    torch::Tensor C = torch::empty({M, K}, options);
    matrix_multiplication_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    return C;
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
}