#include "utils.cuh"

#include <torch/extension.h>


static __device__ __forceinline__ void fused_adamw_update(
    float & param,
    float & mm,
    float & vv,
    const float grad,
    const float beta1,
    const float beta2,
    const float lr,
    const float keep,
    float inv_beta1,
    float inv_beta2,
    const float eps,
    const int t) {
    
    // Update biased first moment estimate
    // param = param * (1 - lr * weight_decay) 
    param *= keep;

    mm = beta1 * mm + (1.0f - beta1) * grad;
    vv = beta2 * vv + (1.0f - beta2) * grad * grad;

    float m_hat = mm * inv_beta1;
    float v_hat = vv * inv_beta2;

    param -= lr * m_hat / (sqrtf(v_hat) + eps);
}


static __global__ void fused_adamw_kernel(
    float* params,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float keep,
    float inv_beta1,
    float inv_beta2,
    int t) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float param = params[i];
        float mm = m[i];
        float vv = v[i];
        fused_adamw_update(param, mm, vv, grad[i], beta1, beta2, lr, keep, inv_beta1, inv_beta2, eps, t);
        params[i] = param;
        m[i] = mm;
        v[i] = vv;
    }
}


static __global__ void fused_adamw_vec4_kernel(
    float* params,
    const float* grad,
    float* m,
    float* v,
    int N,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float keep,
    float inv_beta1,
    float inv_beta2,
    int t)
{
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < (N >> 2); i += blockDim.x * gridDim.x) {
        float4 param = reinterpret_cast<float4*>(params)[i];
        float4 mm = reinterpret_cast<float4*>(m)[i];
        float4 vv = reinterpret_cast<float4*>(v)[i];
        float4 g = reinterpret_cast<const float4*>(grad)[i];

        fused_adamw_update(param.x, mm.x, vv.x, g.x, beta1, beta2, lr, keep, inv_beta1, inv_beta2, eps, t);
        fused_adamw_update(param.y, mm.y, vv.y, g.y, beta1, beta2, lr, keep, inv_beta1, inv_beta2, eps, t);
        fused_adamw_update(param.z, mm.z, vv.z, g.z, beta1, beta2, lr, keep, inv_beta1, inv_beta2, eps, t);
        fused_adamw_update(param.w, mm.w, vv.w, g.w, beta1, beta2, lr, keep, inv_beta1, inv_beta2, eps, t);

        reinterpret_cast<float4*>(params)[i] = param;
        reinterpret_cast<float4*>(m)[i] = mm;
        reinterpret_cast<float4*>(v)[i] = vv;
    }
}


// params, grad, m, v are device pointers
void fused_adamw_cuda(float* params, const float* grad, float* m, float* v, int N, float lr,
    float beta1, float beta2, float eps, float weight_decay, int t) 
{
    
    const float keep = 1.0f - lr * weight_decay;
    const float inv_beta1 = 1.0f / (1.0f - powf(beta1, t));
    const float inv_beta2 = 1.0f / (1.0f - powf(beta2, t));

    if (N % 4 == 0) {
        const int threads = 256;
        const int blocks = ((N >> 2) + threads - 1) / threads;
        fused_adamw_vec4_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1, beta2, eps, keep, inv_beta1, inv_beta2, t);
    } else {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        fused_adamw_kernel<<<blocks, threads>>>(params, grad, m, v, N, lr, beta1, beta2, eps, keep, inv_beta1, inv_beta2, t);
    }
}


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
    int t) 
{
    
    const int N = params.numel();
    
    fused_adamw_cuda(
        params.data_ptr<float>(), 
        grad.data_ptr<float>(), 
        m.data_ptr<float>(), 
        v.data_ptr<float>(),
        N, lr, beta1, beta2, eps, weight_decay, t);
}

