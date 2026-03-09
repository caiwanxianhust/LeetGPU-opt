#include "utils.cuh"
#include "cuda_kernels.cuh"


/*
https://leetgpu.com/challenges/gpt-2-transformer-block
*/


namespace {
    // parameters for GPT-2 small
    constexpr int NUM_HEADS = 12;
    constexpr int D_MODEL = 768;
    constexpr int SIZE_PER_HEAD = D_MODEL / NUM_HEADS;
    constexpr int FFN_DIM = 3072;
    constexpr float EPS = 1e-5f;
    constexpr int TILE_SIZE = 32;

    // Offsets for weights and biases in the weights array
    constexpr int OFF_GAMMA_1 = 0;
    constexpr int OFF_BETA_1 = 768;
    constexpr int OFF_W_QKV = 1536;
    constexpr int OFF_BAIS_QKV = 1771008;
    constexpr int OFF_W_ATTN = 1773312;
    constexpr int OFF_BAIS_ATTN = 2363136;
    constexpr int OFF_GAMMA_2 = 2363904;
    constexpr int OFF_BETA_2 = 2364672;
    constexpr int OFF_W_FC = 2365440;
    constexpr int OFF_BAIS_FC = 4724736;
    constexpr int OFF_W_PROJ = 4727808;
    constexpr int OFF_BAIS_PROJ = 7087104;


    __host__ __device__ inline int div_up(int a, int b) {
        return (a + b - 1) / b;
    }
}





// each warp processes one row(768 features) of the input
__global__ void layer_norm_kernel(const float* x, const float* gamma, const float* beta, float* output, int seq_len) 
{
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_id = blockIdx.x * 4 + warp_id; // 每个 block 4 个 warp，分别处理 4 行
    if (row_id >= seq_len) return;

    float val_mean = 0.0f;
    float val_var = 0.0f;

    // mean
    for (int i = lane_id; i < D_MODEL; i += 32) {
        val_mean += x[row_id * D_MODEL + i];
    }
    __syncthreads();

    val_mean = warp_all_reduce_sum(val_mean);
    val_mean /= (float)D_MODEL;

    // variance
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        val_var += (val - val_mean) * (val - val_mean);
    }
    __syncthreads();

    val_var = warp_all_reduce_sum(val_var);
    val_var /= (float)D_MODEL;

    // output
    for (int i = lane_id; i < D_MODEL; i += 32) {
        float val = x[row_id * D_MODEL + i];
        output[row_id * D_MODEL + i] = (val - val_mean) / sqrtf(val_var + EPS) * gamma[i] + beta[i];
    }
}


void layer_norm(const float* x, const float* gamma, const float* beta, float* output, int seq_len) 
{ 
    int block_size = 128;
    int grid_size = div_up(seq_len, 4); // one warp per row
    layer_norm_kernel<<<grid_size, block_size>>>(x, gamma, beta, output, seq_len);
}


/**
 * 矩阵乘法
 * @param C 输出矩阵，行主序
 * @param A 输入矩阵，行主序
 * @param B 输入矩阵，行主序
 * @param lda 输入矩阵的主维度
 * @param ldb 输入矩阵的主维度
 * @param ldc 输出矩阵的主维度
 */
static __device__ void mma_m16n16k16_fp32(float* C, const float* A, const float* B, int lda, int ldb, int ldc) {
    const int lane_id = threadIdx.x % 32;
    int row_c = lane_id / 8;
    int col_c = lane_id % 8;
    for (int row = row_c; row < 16; row += 4) { 
        for (int col = col_c; col < 16; col += 8) {
            float acc = 0.0f;
            float a_val, b_val;
            #pragma unroll
            for (int k = 0; k < 16; k++) {
                a_val = (A[row * lda + k]);
                b_val = (B[k * ldb + col]);
                acc += a_val * b_val;
            }
            C[row * ldc + col] = acc + C[row * ldc + col];
            __syncwarp();
        }
    }
}

template < int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
__global__ void matmul_bias_warp_kernel(const float* A, const float* B, const float* Bias, float* C, int M, int N, int K) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ float s_a[BM * lda];
    __shared__ float s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    // warp 排布 2x2
    constexpr int warp_col = 2;
    constexpr int warp_row = 2;
    // assert(warp_col * warp_row == blockDim.x / 32);

    // warp 在 BM 和 BN 轴上分别循环 2 次
    constexpr int M_LOOP = BM / (warp_row * MMA_M);
    constexpr int N_LOOP = BN / (warp_col * MMA_N);
    // warp 在 BK 轴上循环 2 次
    constexpr int K_LOOP = BK / (MMA_K);

    // 当前 warp 的索引
    const int warp_col_id = warp_id % (warp_col);
    const int warp_row_id = warp_id / (warp_col);

    // initialize C in shared by zero
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : (0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : (0.0f);
        }
        __syncthreads();

        // compute block mma 64 x 64
        for (int m_id = 0; m_id < M_LOOP; m_id++) {
            for (int n_id = 0; n_id < N_LOOP; n_id++) {
                int a_off = m_id * (MMA_M * warp_row) * lda + warp_row_id * MMA_M * lda;
                int b_off = n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                int c_off = m_id * (MMA_M * warp_row) * ldc + warp_row_id * MMA_M * ldc + n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                for (int k_id = 0; k_id < K_LOOP; k_id++) {
                    a_off += k_id * MMA_K;
                    b_off += k_id * MMA_K * ldb;
                    mma_m16n16k16_fp32(s_c + c_off, s_a + a_off, s_b + b_off, lda, ldb, ldc);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = s_c[row * ldc + col] + Bias[col_c];
        }
    }
}

void matmul_bias_warp_fp32(const float* A, const float* B, const float* Bias, float* C, int M, int N, int K) { 
    // 每个 warp 计算一个 16x16 的子矩阵，总共 4 个 warp 计算一个 32x32 的子矩阵
    // 每个 block 计算一个 64 x 64 的子矩阵，所以 4 个 warp 分别在 M 和 N 方向上循环 2 次
    // BK = 32, 每个 warp 计算 16x16 矩阵时，需要在 K 维度上计算两次 mma 并累加
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int BK = 32;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int block_size = 128; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_bias_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, Bias, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


template < int BM, int BN, int BK, int MMA_M, int MMA_N, int MMA_K>
__global__ void matmul_bias_res_warp_kernel(const float* A, const float* B, const float* Bias, const float* front, float* C, int M, int N, int K) { 
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // 为了避免 bank conflict，增加 padding
    const int lda = BK; 
    const int ldb = BN;
    const int ldc = BN;
    __shared__ float s_a[BM * lda];
    __shared__ float s_b[BK * ldb];
    __shared__ float s_c[BM * ldc];

    // warp 排布 2x2
    constexpr int warp_col = 2;
    constexpr int warp_row = 2;
    // assert(warp_col * warp_row == blockDim.x / 32);

    // warp 在 BM 和 BN 轴上分别循环 2 次
    constexpr int M_LOOP = BM / (warp_row * MMA_M);
    constexpr int N_LOOP = BN / (warp_col * MMA_N);
    // warp 在 BK 轴上循环 2 次
    constexpr int K_LOOP = BK / (MMA_K);

    // 当前 warp 的索引
    const int warp_col_id = warp_id % (warp_col);
    const int warp_row_id = warp_id / (warp_col);

    // initialize C in shared by zero
    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;;
        int col = i % BN;
        s_c[row * ldc + col] = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        // load A from global to shared
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            int row_a = i / BK + blockIdx.y * BM;
            int col_a = i % BK + k;
            s_a[row * lda + col] = (row_a < M && col_a < K) ? A[row_a * K + col_a] : (0.0f);
        }
        // load B from global to shared
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            int row_b = i / BN + k;
            int col_b = i % BN + blockIdx.x * BN;
            s_b[row * ldb + col] = (row_b < K && col_b < N) ? B[row_b * N + col_b] : (0.0f);
        }
        __syncthreads();

        // compute block mma 64 x 64
        for (int m_id = 0; m_id < M_LOOP; m_id++) {
            for (int n_id = 0; n_id < N_LOOP; n_id++) {
                int a_off = m_id * (MMA_M * warp_row) * lda + warp_row_id * MMA_M * lda;
                int b_off = n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                int c_off = m_id * (MMA_M * warp_row) * ldc + warp_row_id * MMA_M * ldc + n_id * (MMA_N * warp_col) + warp_col_id * MMA_N;
                for (int k_id = 0; k_id < K_LOOP; k_id++) {
                    a_off += k_id * MMA_K;
                    b_off += k_id * MMA_K * ldb;
                    mma_m16n16k16_fp32(s_c + c_off, s_a + a_off, s_b + b_off, lda, ldb, ldc);
                }
            }
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < BM * BN; i += blockDim.x) {
        int row = i / BN;
        int col = i % BN;
        int row_c = i / BN + blockIdx.y * BM;
        int col_c = i % BN + blockIdx.x * BN;
        if (row_c < M && col_c < N) {
            C[row_c * N + col_c] = s_c[row * ldc + col] + Bias[col_c] + front[row_c * N + col_c];
        }
    }
}

void matmul_bias_res_warp_fp32(const float* A, const float* B, const float* Bias, const float* front, float* C, int M, int N, int K) { 
    // 每个 warp 计算一个 16x16 的子矩阵，总共 4 个 warp 计算一个 32x32 的子矩阵
    // 每个 block 计算一个 64 x 64 的子矩阵，所以 4 个 warp 分别在 M 和 N 方向上循环 2 次
    // BK = 32, 每个 warp 计算 16x16 矩阵时，需要在 K 维度上计算两次 mma 并累加
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 16;
    constexpr int MMA_K = 16;
    constexpr int BK = 32;
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int block_size = 128; // 每个 block 含有多少线程数
    dim3 block(block_size);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    matmul_bias_res_warp_kernel<BM, BN, BK, MMA_M, MMA_N, MMA_K><<<grid, block>>>(A, B, Bias, front, C, M, N, K);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


/**
 * @brief    Multi-head QK scaled
 * @param    Q, K, qk are device pointers, Q[N, h, size_per_head], K[N, h, size_per_head], qk[h, N, N]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void qk_scaled_kernel(const float* Q, const float*K, float* qk, int N, int size_per_head, int d_model) 
{
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    int offset_c = head_id * N * N;
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_q[TILE_SIZE * TILE_SIZE];
    __shared__ float s_k[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < size_per_head; k += TILE_SIZE) {
        // load Q and K to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_q[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? Q[row * d_model + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.x * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE + head_id * size_per_head;
            s_k[i] = (row < N && (k + i % TILE_SIZE) < size_per_head && col < d_model) ? K[row * d_model + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_q[ty * TILE_SIZE + i] * s_k[tx * TILE_SIZE + i];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < N) {
        qk[offset_c + row_c * N + col_c] = acc / sqrtf((float)size_per_head);
    }
}


/**
 * @brief    Softmax
 * @param    qk, attn_scores are device pointers, qk[h, N, N], attn_scores[h, N, N]
 * @param    N sequence length
 * @return   void
 */
static __global__ void softmax_kernel(const float* qk, float* attn_scores, int N) 
{
    float max_val = -INFINITY;
    float exp_sum = 0.0f;
    float val;

    // Find max
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        max_val = max(max_val, val);
    }
    __syncthreads();
    max_val = block_all_reduce_max(max_val);

    // Compute exp sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        exp_sum += expf(val - max_val);
    }
    __syncthreads();
    exp_sum = block_all_reduce_sum(exp_sum);

    // Write output
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        val = qk[blockIdx.x * N + i];
        attn_scores[blockIdx.x * N + i] = expf(val - max_val) / exp_sum;
    }
}


/**
 * @brief    Multi-head attention
 * @param    attn_scores, V, output are device pointers, attn_scores[h, N, N], V[N, h, size_per_head], output[N, h, size_per_head]
 * @param    N sequence length
 * @param    size_per_head size per head
 * @return   void
 */
__global__ void multi_head_attention_kernel(const float* attn_scores, const float* V, float* output, int N, int size_per_head, int d_model) 
{ 
    int head_id = blockIdx.z;
    int tx = threadIdx.x % TILE_SIZE;
    int ty = threadIdx.x / TILE_SIZE;
    
    int row_c = blockIdx.y * TILE_SIZE + ty;
    int col_c = blockIdx.x * TILE_SIZE + tx;

    __shared__ float s_attn[TILE_SIZE * TILE_SIZE];
    __shared__ float s_v[TILE_SIZE * TILE_SIZE];

    float acc = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        // load attn_scores and V to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = blockIdx.y * TILE_SIZE + i / TILE_SIZE;
            int col = k + i % TILE_SIZE;
            s_attn[i] = (row < N && col < N) ? attn_scores[head_id * N * N + row * N + col] : 0.0f;
        }

        for (int i = threadIdx.x; i < TILE_SIZE * TILE_SIZE; i += blockDim.x) {
            int row = k + i / TILE_SIZE;
            int col = blockIdx.x * TILE_SIZE + i % TILE_SIZE;
            s_v[i] = (row < N && col < size_per_head) ? V[row * d_model + head_id * size_per_head + col] : 0.0f;
        }
        __syncthreads();

        if (row_c < N && col_c < size_per_head) {
            for (int i = 0; i < TILE_SIZE; i++) {
                acc += s_attn[ty * TILE_SIZE + i] * s_v[i * TILE_SIZE + tx];
            }
        }
        __syncthreads();
    }
    if (row_c < N && col_c < size_per_head) {
        output[row_c * d_model + col_c + head_id * size_per_head] = acc;
    }
}




// Q, K, V, output are device pointers
void multi_head_attention(const float* Q, const float* K, const float* V, float* qk, float* attn_scores, float* attn_out, int N) 
{
    
    // QK^T / sqrt(d_model)
    dim3 block(TILE_SIZE * TILE_SIZE);
    dim3 grid(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE), NUM_HEADS);
    qk_scaled_kernel<<<grid, block>>>(Q, K, qk, N, SIZE_PER_HEAD, D_MODEL);

    // Softmax
    softmax_kernel<<<NUM_HEADS * N, 256>>>(qk, attn_scores, N);

    // attn_scores * V
    grid = dim3(div_up(SIZE_PER_HEAD, TILE_SIZE), div_up(N, TILE_SIZE), NUM_HEADS);
    multi_head_attention_kernel<<<grid, block>>>(attn_scores, V, attn_out, N, SIZE_PER_HEAD, D_MODEL);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

static __global__ void gelu_kernel(const float* input, float* output, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    }
}

static void gelu(const float* input, float* output, int size) 
{
    int block_size = 256;
    int grid_size = div_up(size, block_size);
    gelu_kernel<<<grid_size, block_size>>>(input, output, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}



// x, output, weights are device pointers
void gpt2_transformer_block_cuda(const float* x, float* output, const float* weights, int seq_len) 
{
    // Load weights and biases from the weights array
    const float* gamma_1 = weights + OFF_GAMMA_1;
    const float* beta_1 = weights + OFF_BETA_1;
    const float* w_qkv = weights + OFF_W_QKV;
    const float* bias_qkv = weights + OFF_BAIS_QKV;
    const float* w_attn = weights + OFF_W_ATTN;
    const float* bias_attn = weights + OFF_BAIS_ATTN;
    const float* gamma_2 = weights + OFF_GAMMA_2;
    const float* beta_2 = weights + OFF_BETA_2;
    const float* w_fc = weights + OFF_W_FC;
    const float* bias_fc = weights + OFF_BAIS_FC;
    const float* w_proj = weights + OFF_W_PROJ;
    const float* bias_proj = weights + OFF_BAIS_PROJ;

    // Allocate intermediate buffers
    float* d_buf; // buffer for intermediate results
    float* x_norm_1; // normalized output after first layer norm
    float* qkv; // concatenated Q, K, V
    float* Q;
    float* K;
    float* V;
    float* qk;
    float* attn_scores;
    float* attn_out;
    float* attn_projected_res;
    float* x_norm_2;
    float* fc_linear;
    float* fc_gelu;

    int buf_size = seq_len * D_MODEL * sizeof(float) +                                      // x_norm_1
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // qkv
    3 * seq_len * D_MODEL * sizeof(float) +                                                 // Q, K, V
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // qk       
    NUM_HEADS * seq_len * seq_len * sizeof(float) +                                         // attn_scores
    seq_len * D_MODEL * sizeof(float) +                                                      // attn_out
    seq_len * D_MODEL * sizeof(float) +                                                       // attn_projected
    seq_len * D_MODEL * sizeof(float) +                                                     // x_norm_2
    seq_len * FFN_DIM * sizeof(float) +                                                           // fc_linear
    seq_len * FFN_DIM * sizeof(float);                                                            // fc_gelu

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    x_norm_1 = d_buf;
    qkv = x_norm_1 + seq_len * D_MODEL;
    Q = qkv + seq_len * 3 * D_MODEL;
    K = Q + seq_len * D_MODEL;
    V = K + seq_len * D_MODEL;
    qk = V + seq_len * D_MODEL;
    attn_scores = qk + NUM_HEADS * seq_len * seq_len;
    attn_out = attn_scores + NUM_HEADS * seq_len * seq_len;
    attn_projected_res = attn_out + seq_len * D_MODEL;
    x_norm_2 = attn_projected_res + seq_len * D_MODEL;
    fc_linear = x_norm_2 + seq_len * D_MODEL;
    fc_gelu = fc_linear + seq_len * FFN_DIM;


    // Step 1: Layer Normalization
    // x(N, D_MODEL) -> x_norm(N, D_MODEL)
    layer_norm(x, gamma_1, beta_1, x_norm_1, seq_len);

    // Step 2: matmul and add bias for QKV
    // x_norm(N, D_MODEL) * w_qkv(D_MODEL, 3 * D_MODEL) + bias_qkv(3 * D_MODEL) -> qkv(N, 3 * D_MODEL)
    matmul_bias_warp_fp32(x_norm_1, w_qkv, bias_qkv, qkv, seq_len, 3 * D_MODEL, D_MODEL);

    // step 3: split QKV
    // qkv(N, 3 * D_MODEL) -> Q(N, D_MODEL), K(N, D_MODEL), V(N, D_MODEL)
    CHECK_CUDA_ERROR(cudaMemcpy2D(Q, D_MODEL * sizeof(float), qkv,               3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy2D(K, D_MODEL * sizeof(float), qkv + D_MODEL,     3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy2D(V, D_MODEL * sizeof(float), qkv + 2 * D_MODEL, 3 * D_MODEL * sizeof(float), D_MODEL * sizeof(float), seq_len, cudaMemcpyDeviceToDevice));

    // step 4: compute attention
    // QK^T / sqrt(d_model) -> attn_scores(N, N)
    // attn_scores(N, N) * V(N, D_MODEL) -> attn_out(N, D_MODEL)
    multi_head_attention(Q, K, V, qk, attn_scores, attn_out, seq_len);
    // attn projection
    // attn_out(N, D_MODEL) * w_attn(D_MODEL, D_MODEL) + bias_attn(D_MODEL) + x(N, D_MODEL) -> attn_projected(N, D_MODEL)
    matmul_bias_res_warp_fp32(attn_out, w_attn, bias_attn, x, attn_projected_res, seq_len, D_MODEL, D_MODEL);

    // step 5: Layer Normalization
    layer_norm(attn_projected_res, gamma_2, beta_2, x_norm_2, seq_len);

    // step 6: FFN
    // x_norm_2(N, D_MODEL) * w_fc(D_MODEL, FFN_DIM) + bias_fc(FFN_DIM) -> fc_linear(N, FFN_DIM)
    matmul_bias_warp_fp32(x_norm_2, w_fc, bias_fc, fc_linear, seq_len, FFN_DIM, D_MODEL);
    
    // step 7: GeLU
    // fc_linear(N, FFN_DIM) -> fc_gelu(N, FFN_DIM)
    gelu(fc_linear, fc_gelu, seq_len * FFN_DIM);

    // step 8: FFN projection
    // fc_gelu(N, FFN_DIM) * w_proj(FFN_DIM, D_MODEL) + bias_proj(D_MODEL) + attn_projected(N, D_MODEL) -> output(N, D_MODEL)
    matmul_bias_res_warp_fp32(fc_gelu, w_proj, bias_proj, attn_projected_res, output, seq_len, D_MODEL, FFN_DIM);

    CHECK_CUDA_ERROR(cudaFree(d_buf));
}