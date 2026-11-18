#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <torch/extension.h>


/*
def pytorch_distillation_kl_divergence(teacher_logits, student_logits, temperature):
        T = float(temperature)
        a = teacher_logits.div(T)          # 新张量，不污染输入
        b = student_logits.div(T)
        diff = a.sub(b)                    # a - b

        amax = a.amax(dim=-1, keepdim=True)
        bmax = b.amax(dim=-1, keepdim=True)

        a.sub_(amax).exp_()                # a := e^{a-amax} ∈ (0,1]
        b.sub_(bmax).exp_()
        sa = a.sum(dim=-1)                 # Σ e^{a-amax}
        sb = b.sum(dim=-1)

        a.mul_(diff)                       # a := e^{a-amax}·(a-b)
        num = a.sum(dim=-1).div_(sa)       # Σⱼ pⱼ·(aⱼ-bⱼ)
        lse_a = sa.log_().add_(amax.squeeze(-1))
        lse_b = sb.log_().add_(bmax.squeeze(-1))
        kl = num.sub_(lse_a).add_(lse_b)
        return kl
*/


static __global__ void distillation_kl_divergence(
    const float* teacher_logits, const float* student_logits, float* kl, int N, int V, float inverse_temperature) 
{
    const float* teacher_row = teacher_logits + blockIdx.x * V;
    const float* student_row = student_logits + blockIdx.x * V;
    float* kl_row = kl + blockIdx.x;

    float amax = -FLT_MAX;
    float bmax = -FLT_MAX;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        amax = fmaxf(amax, teacher_row[i] * inverse_temperature);
        bmax = fmaxf(bmax, student_row[i] * inverse_temperature);
    }
    __syncthreads();
    amax = block_all_reduce_max(amax);
    bmax = block_all_reduce_max(bmax);

    float a_expsum = 0.0f;
    float b_expsum = 0.0f;
    float num = 0.0f;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        float a = teacher_row[i] * inverse_temperature;
        float b = student_row[i] * inverse_temperature;
        float diff = a - b;
        float ea = expf(a - amax);
        float eb = expf(b - bmax);
        a_expsum += ea;
        b_expsum += eb;
        num += ea * diff;
    }
    __syncthreads();
    a_expsum = block_all_reduce_sum(a_expsum);
    b_expsum = block_all_reduce_sum(b_expsum);
    num = block_all_reduce_sum(num);
    num /= a_expsum;

    float lse_a = logf(a_expsum) + amax;
    float lse_b = logf(b_expsum) + bmax;
    float kl_value = num - lse_a + lse_b;

    if (threadIdx.x == 0) {
        kl_row[0] = kl_value;
    }
}


/**
    * Compute the KL divergence between teacher and student logits.
    *
    * @param teacher_logits: [N, V] tensor of teacher logits (device memory).
    * @param student_logits: [N, V] tensor of student logits (device memory).
    * @param kl: [N] tensor to store the output KL divergence (device memory).
    * @param N: Number of samples.
    * @param V: Vocabulary size (number of classes).
*/
// teacher_logits, student_logits, kl are device pointers
void distillation_kl_divergence_cuda(
    const float* teacher_logits, const float* student_logits, float* kl, int N, int V, float temperature) 
{
    
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    float inverse_temperature = 1.0f / temperature;

    distillation_kl_divergence<<<blocksPerGrid, threadsPerBlock>>>(
        teacher_logits, student_logits, kl, N, V, inverse_temperature);
}



torch::Tensor distillation_kl_divergence(
    const torch::Tensor& teacher_logits, const torch::Tensor& student_logits, float temperature) 
{
    TORCH_CHECK(teacher_logits.dim() == 2, "teacher_logits must be a 2D tensor");
    TORCH_CHECK(student_logits.dim() == 2, "student_logits must be a 2D tensor");
    TORCH_CHECK(teacher_logits.size(0) == student_logits.size(0), "teacher and student logits must have the same batch size");
    TORCH_CHECK(teacher_logits.size(1) == student_logits.size(1), "teacher and student logits must have the same number of classes");

    int N = teacher_logits.size(0);
    int V = teacher_logits.size(1);

    auto kl = torch::empty({N}, teacher_logits.options());

    distillation_kl_divergence_cuda(
        teacher_logits.data_ptr<float>(), 
        student_logits.data_ptr<float>(), 
        kl.data_ptr<float>(), 
        N, V, temperature);

    return kl;
}