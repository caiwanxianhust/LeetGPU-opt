#include "utils.cuh"

#include <math_constants.h>
#include <torch/extension.h>


namespace {
    constexpr int PROMPT_LEN = 31;
    constexpr int DECODE_STEPS = 11;
    constexpr int VOCAB = 10;
    constexpr int MAX_LEN = 42;
    constexpr float EPS = 1e-6f;
    constexpr float OMEGA = 2.0f * CUDART_PI_F / 19.0f;
}


static __device__ __forceinline__ void unit_rms(float &x0, float &x1) {
    float m = 0.5f * (x0 * x0 + x1 * x1) + EPS;
    float inv = rsqrtf(m);
    x0 *= inv;
    x1 *= inv;
}

static __device__ __forceinline__ void rope(float &x0, float &x1, int pos) {
    float ang = OMEGA * (float)pos;
    float s, c;
    __sincosf(ang, &s, &c);

    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;
    x0 = y0;
    x1 = y1;
}

static __device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

static __global__ void adder_kernel(const int* __restrict__ prompts,
                             float* __restrict__ output,
                             const float* __restrict__ weights,
                             int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    const float w0 = weights[0];
    const float w1 = weights[1];

    const float q0 = weights[2];
    const float q1 = weights[3];

    const float v0 = weights[4];

    const float a = weights[5];
    const float c = weights[6];
    const float carry = weights[7];

    const float n0 = weights[8];
    const float n1 = weights[9];

    float s2 = logf(10.0f) /
               (sqrtf(2.0f) * (cosf(0.3f * OMEGA) - cosf(0.7f * OMEGA)));
    float attn_scale = rsqrtf(2.0f) * s2;

    int tokens[MAX_LEN];

    #pragma unroll
    for (int i = 0; i < PROMPT_LEN; ++i) {
        tokens[i] = prompts[b * PROMPT_LEN + i];
    }

    int cur_len = PROMPT_LEN;

    for (int step = 0; step < DECODE_STEPS; ++step) {
        int pos = cur_len - 1;
        int d_last = tokens[pos];

        float h0 = w0 - w1 * (float)(d_last * d_last);
        float h1 = -(float)d_last;

        float qh0 = h0;
        float qh1 = h1;
        unit_rms(qh0, qh1);

        float Q0 = qh0 * q0;
        float Q1 = qh0 * q1;
        unit_rms(Q0, Q1);
        rope(Q0, Q1, pos);

        float scores[MAX_LEN];

        float max_score = -INFINITY;

        for (int j = 0; j < cur_len; ++j) {
            int dj = tokens[j];

            float eh0 = w0 - w1 * (float)(dj * dj);
            float eh1 = -(float)dj;

            unit_rms(eh0, eh1);

            float K0 = eh0;
            float K1 = 0.0f;
            unit_rms(K0, K1);
            rope(K0, K1, j);

            float sc = (Q0 * K0 + Q1 * K1) * attn_scale;
            scores[j] = sc;
            if (sc > max_score) max_score = sc;
        }

        float denom = 0.0f;
        float attn0 = 0.0f;

        for (int j = 0; j < cur_len; ++j) {
            int dj = tokens[j];

            float eh0 = w0 - w1 * (float)(dj * dj);
            float eh1 = -(float)dj;

            unit_rms(eh0, eh1);

            float V0 = eh1 * v0;

            float e = __expf(scores[j] - max_score);
            denom += e;
            attn0 += e * V0;
        }

        attn0 /= denom;

        h1 += attn0;

        float mh0 = h0;
        float mh1 = h1;
        unit_rms(mh0, mh1);

        float g0 = mh0 * a + mh1 * c;
        float g1 = mh0 * (a - c / 1000.0f) + mh1 * c;

        float base = mh0;
        float mix0 = silu(g0) * base;
        float mix1 = silu(g1) * base;

        h1 += carry * (mix1 - mix0);

        float out0 = h0;
        float out1 = h1;
        float rms = rsqrtf(0.5f * (out0 * out0 + out1 * out1) + EPS);
        out0 = out0 * rms * n0;
        out1 = out1 * rms * n1;

        float best_val = -INFINITY;
        int best_digit = 0;

        #pragma unroll
        for (int d = 0; d < VOCAB; ++d) {
            float e0 = w0 - w1 * (float)(d * d);
            float e1 = -(float)d;

            float logit = out0 * e0 + out1 * e1;
            output[(b * DECODE_STEPS + step) * VOCAB + d] = logit;

            if (logit > best_val) {
                best_val = logit;
                best_digit = d;
            }
        }

        tokens[cur_len] = best_digit;
        cur_len++;
    }
}

/**
 * @param prompts: [batch_size, PROMPT_LEN]
 * @param output: [batch_size, DECODE_STEPS, VOCAB]
 * @param weights: [10]
 */
// prompts, output, weights are device pointers
void adder_transformer_inference_cuda(const int* prompts, float* output, const float* weights, int batch_size) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    adder_kernel<<<blocks, threads>>>(prompts, output, weights, batch_size);
}


torch::Tensor adder_transformer_inference(
    const torch::Tensor& prompts,
    const torch::Tensor& weights
) { 
    TORCH_CHECK(prompts.dim() == 2, "prompts must be a 2D tensor");
    TORCH_CHECK(prompts.size(1) == PROMPT_LEN, "prompts must have size PROMPT_LEN");
    TORCH_CHECK(weights.dim() == 1, "weights must be a 1D tensor");
    TORCH_CHECK(weights.size(0) == 10, "weights must have size 10");

    int batch_size = prompts.size(0);

    torch::Tensor output = torch::empty({batch_size, DECODE_STEPS, VOCAB}, weights.options());
    adder_transformer_inference_cuda(
        prompts.data_ptr<int>(), 
        output.data_ptr<float>(), 
        weights.data_ptr<float>(), 
        batch_size);
    return output;    
}

