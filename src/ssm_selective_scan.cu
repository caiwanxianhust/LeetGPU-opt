#include "utils.cuh"

#include <torch/extension.h>


/*
Implement the forward pass of a State Space Model (SSM) selective scan, the core operation in Mamba-style sequence models. Given an input sequence u, time-step parameters delta, state-transition matrix A, input projection B, output projection C, and skip-connection weights skip, compute the output sequence y in float32.

Implementation Requirements
Implement the function solve(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state) with the signature unchanged. Do not use external libraries beyond the allowed framework. Write the result into the pre-allocated output tensor y.


The initial hidden state is 0. All channels d are independent: they share the same B and C projections but have separate state-transition rows in A.

Example
Input:
  u     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]  shape (1,4,2)
  delta = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]]  shape (1,4,2)
  A     = [[-0.5, -1.0], [-0.5, -1.0]]                         shape (2,2)
  B     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
  C     = [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]]  shape (1,4,2)
  skip  = [0.0, 0.0]                                            shape (2,)
  batch=1, seq_len=4, d_model=2, d_state=2

Derivation (delta=1 everywhere, so A_bar_dn = exp(A_dn)):
  A_bar[d=0] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]
  A_bar[d=1] = [exp(-0.5), exp(-1.0)] ≈ [0.607, 0.368]

  Hidden state h has shape (d_model=2, d_state=2); initial h = zeros.
  t=0: h = [[1.000, 0.000], [0.000, 0.000]]  →  y[0,0] = [1.000, 0.000]
  t=1: h = [[0.607, 0.000], [0.000, 1.000]]  →  y[0,1] = [0.000, 1.000]
  t=2: h = [[1.368, 1.000], [1.000, 1.368]]  →  y[0,2] = [2.368, 2.368]
  t=3: h = [[0.830, 0.368], [0.607, 0.503]]  →  y[0,3] = [0.599, 0.555]

Output:
  y = [[[1.000, 0.000], [0.000, 1.000], [2.368, 2.368], [0.599, 0.555]]]
Constraints
1 ≤ batch ≤ 16
1 ≤ seq_len ≤ 8,192
1 ≤ d_model ≤ 2,048
1 ≤ d_state ≤ 64
All entries of delta are positive
All entries of A are negative (ensuring A_bar ∈ (0, 1))
All tensors are float32 on the GPU
Performance is measured with batch = 4, seq_len = 4,096, d_model = 512, d_state = 16

*/



namespace ssm_naive {


__global__ void selective_scan_kernel(
    const float* __restrict__ u,
    const float* __restrict__ delta,
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const float* __restrict__ skip,
    float* __restrict__ y,
    int batch, int seq_len, int d_model, int d_state
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * d_model;

    if (idx >= total) return;

    int channel_id = idx % d_model;          // channel index
    int batch_id = idx / d_model;          // batch index

    float h[64];  

    for (int n = 0; n < d_state; ++n) {
        h[n] = 0.0f;
    }

    const float* A_d = A + channel_id * d_state;
    float skip_d = skip[channel_id];

    for (int t = 0; t < seq_len; ++t) {
        int ud_idx = (batch_id * seq_len + t) * d_model + channel_id;
        int state_idx = (batch_id * seq_len + t) * d_state;

        float u_val = u[ud_idx];
        float delta_val = delta[ud_idx];

        float out = 0.0f;

        for (int n = 0; n < d_state; ++n) {
            float a_bar = expf(delta_val * A_d[n]);
            float b_bar = delta_val * B[state_idx + n];

                h[n] = a_bar * h[n] + b_bar * u_val;

            out += C[state_idx + n] * h[n];
        }

        y[ud_idx] = out + skip_d * u_val;
    }
}

void ssm_selective_scan_naive(const float* u, const float* delta, const float* A, const float* B,
    const float* C, const float* skip, float* y, int batch, int seq_len, int d_model, int d_state
) {
    int total = batch * d_model;

    constexpr int threads = 128;
    int blocks = (total + threads - 1) / threads;

    selective_scan_kernel<<<blocks, threads>>>(
        u, delta, A, B, C, skip, y,
        batch, seq_len, d_model, d_state
    );
}


}   // namespace ssm_naive











/**
 * @brief Compute the forward pass of a State Space Model (SSM) selective scan on the GPU.
 * @param u The input sequence. Shape: (batch, seq_len, d_model)
 * @param delta Time-step parameters. Shape: (batch, seq_len, d_model)
 * @param A State-transition matrix. Shape: (d_model, d_state)
 * @param B Input projection. Shape: (batch, seq_len, d_state)
 * @param C Output projection. Shape: (batch, seq_len, d_state)
 * @param skip Skip-connection weights. Shape: (d_model,)
 * @param y Output sequence. Shape: (batch, seq_len, d_model)
 * @param batch Batch size.
 * @param seq_len Sequence length.
 * @param d_model Model dimension.
 * @param d_state State dimension.
 */
// u, delta, A, B, C, skip, y are device pointers
void ssm_selective_scan_cuda(const float* u, const float* delta, const float* A, const float* B,
    const float* C, const float* skip, float* y, int batch, int seq_len,
    int d_model, int d_state) 
{
    ssm_naive::ssm_selective_scan_naive(u, delta, A, B, C, skip, y, batch, seq_len, d_model, d_state);
}


torch::Tensor ssm_selective_scan(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& skip
) {
    TORCH_CHECK(u.ndimension() == 3, "u must be 3D");
    TORCH_CHECK(delta.ndimension() == 3, "delta must be 3D");
    TORCH_CHECK(A.ndimension() == 2, "A must be 2D");
    TORCH_CHECK(B.ndimension() == 3, "B must be 3D");
    TORCH_CHECK(C.ndimension() == 3, "C must be 3D");
    TORCH_CHECK(skip.ndimension() == 1, "skip must be 1D");

    int batch = u.size(0);
    int seq_len = u.size(1);
    int d_model = u.size(2);
    int d_state = A.size(1);

    TORCH_CHECK(delta.size(0) == batch && delta.size(1) == seq_len && delta.size(2) == d_model, "delta must have the same shape as u");
    TORCH_CHECK(A.size(0) == d_model, "A must have shape (d_model, d_state)");
    TORCH_CHECK(B.size(0) == batch && B.size(1) == seq_len && B.size(2) == d_state, "B must have shape (batch, seq_len, d_state)");
    TORCH_CHECK(C.size(0) == batch && C.size(1) == seq_len && C.size(2) == d_state, "C must have shape (batch, seq_len, d_state)");
    TORCH_CHECK(skip.size(0) == d_model, "skip must have shape (d_model)");

    torch::Tensor output = torch::empty_like(u);
    ssm_selective_scan_cuda(
        u.data_ptr<float>(),
        delta.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        skip.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, seq_len, d_model, d_state
    );
    return output;
}