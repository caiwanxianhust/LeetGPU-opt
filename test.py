import torch
import leetgpu
import time


def matrix_multiplication():
    # 测试FP32
    a = torch.randn(1024, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 512, device="cuda", dtype=torch.float32)

    start_time = time.time()
    c = leetgpu.matmul(a, b)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    print(a[:4, :4])
    print(b[:4, :4])
    print(c[:4, :4])
    print(torch.matmul(a, b)[:4, :4])

    assert torch.allclose(c, torch.matmul(a, b), atol=1e-3)
    print("All tests passed!")


def vector_addition():
    # 测试FP32
    a = torch.randn(1024 * 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024 * 1024, device="cuda", dtype=torch.float32)

    start_time = time.time()
    c = leetgpu.vector_add(a, b)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    print(a[:10])
    print(b[:10])
    print(c[:10])
    print((a + b)[:10])

    assert torch.allclose(c, a + b, atol=1e-5)
    print("All tests passed!")


def conv1D():
    # 测试FP32
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    w = torch.randn(2, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.conv1d(x.view(1, 1, -1), w.view(1, 1, -1)).view(-1)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.conv1D(x, w)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(w[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def reverse_array():
    # 测试FP32
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    print(x[:10])

    # 用 PyTorch 实现
    torch_out = torch.flip(x, dims=[0])

    # 用 LeetGPU 实现
    start_time = time.time()
    leetgpu.reverse_array(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(torch_out[:10])

    assert torch.allclose(x, torch_out, atol=1e-3)
    print("All tests passed!")


def transpose():
    # 测试FP32
    x = torch.randn(1024, 898, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.transpose(x, 0, 1)
    
    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.mat_transpose(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:4, :4])
    print(out[:4, :4])
    print(torch_out[:4, :4])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def color_inversion():
    # 测试
    x = torch.randint(0, 256, (512, 1024, 4), device="cuda", dtype=torch.uint8)

    # 用 PyTorch 实现
    torch_out = torch.empty_like(x, device="cuda", dtype=torch.uint8)
    torch_out[:, :, 0] = 255 - x[:, :, 0]
    torch_out[:, :, 1] = 255 - x[:, :, 1]
    torch_out[:, :, 2] = 255 - x[:, :, 2]
    torch_out[:, :, 3] = x[:, :, 3]
    print("x: ")
    print(x[:4, :4, :4])
    print("pytorch out: ")
    print(torch_out[:4, :4, :4])

    inp = x.flatten()
    print("input: ")
    print(inp[:16])
    
    # 用 LeetGPU 实现
    start_time = time.time()
    leetgpu.color_invert(inp, 512, 1024)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("cuda out: ")
    print(inp[:16])
    print("pytorch out: ")
    print(torch_out.flatten()[:16])

    assert torch.allclose(inp, torch_out.flatten(), atol=1e-3)
    print("All tests passed!")


def relu():
    # 测试
    x = torch.randn(1025, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.relu(x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.relu(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def leaky_relu():
    # 测试
    x = torch.randn(1025, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.leaky_relu(x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.leaky_relu(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def rainbow_table():
    # 使用 randint 创建 int32 类型的张量
    x = torch.randint(-2147483648, 2147483647, (1025,), device="cuda", dtype=torch.int32)
    R = 4

    # 用 PyTorch 实现
    def fnv1a_hash(inp):
        FNV_PRIME = 16777619;
        OFFSET_BASIS = 2166136261;
        
        hash = OFFSET_BASIS;
        
        for byte_pos in range(4): 
            byte = (inp >> (byte_pos * 8)) & 0xff;
            hash = (hash ^ byte) * FNV_PRIME;
        return hash
    
    torch_out = torch.empty_like(x, device="cuda", dtype=torch.uint32)
    for i in range(x.shape[0]):
        val = x[i]
        for j in range(R):
            val = fnv1a_hash(val)
        torch_out[i] = val
    
    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.rainbow_table(x, R)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def matrix_copy():
    # 测试
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)

    # 用 leetgpu 实现
    start_time = time.time()
    b = leetgpu.matrix_copy(a.flatten(), 1024)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(a.flatten()[:16])
    print(b[:16])

    # 检查结果
    assert torch.allclose(b, a.flatten(), atol=1e-5)
    print("All tests passed!")


def count_array_elements():
    # 测试
    x = torch.randint(-127, 127, (1025,), device="cuda", dtype=torch.int32)
    K = 0

    # 用 pytorch 实现
    torch_out = (x == K).sum().item()

    # 用 leetgpu 实现
    start_time = time.time()
    out = leetgpu.count_equal(x, K)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out.item())
    print(torch_out)

    # 检查结果
    assert out.item() == torch_out
    print("All tests passed!")


def count_array_elements_2D():
    # 测试
    x = torch.randint(-127, 127, (1025, 2049), device="cuda", dtype=torch.int32)
    K = 0

    # 用 pytorch 实现
    torch_out = (x == K).sum().item()

    # 用 leetgpu 实现
    start_time = time.time()
    out = leetgpu.count_equal_2D(x.flatten(), 1025, 2049, K)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out.item())
    print(torch_out)

    # 检查结果
    assert out.item() == torch_out
    print("All tests passed!")


def silu():
    # 测试
    x = torch.randn(1025, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.silu(x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.silu(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def swiglu():
    # 测试
    x = torch.randn(2500, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.silu(x[:1250]) * x[1250:]

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.swiglu(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-5)
    print("All tests passed!")


def reduce_sum():
    # 测试
    x = torch.randn(1024 * 1024, device="cuda", dtype=torch.float32)


    # 用 PyTorch 实现
    torch_out = torch.sum(x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.reduce_sum(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out.item())
    print(torch_out.item())

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def softmax_1D():
    # 测试
    x = torch.randn(1024 * 128, device="cuda", dtype=torch.float32)
    # x = torch.arange(0, 10, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.softmax(x, dim=0)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.softmax_1D(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def softmax_attention():
    # 测试
    Q = torch.randn(512, 200, device="cuda", dtype=torch.float32)
    K = torch.randn(256, 200, device="cuda", dtype=torch.float32)
    V = torch.randn(256, 200, device="cuda", dtype=torch.float32)
    
    # 用 PyTorch 实现
    scores = torch.matmul(Q, K.t()) / (200 ** 0.5)
    print(scores[:2, :4])
    attn_weights = torch.softmax(scores, dim=-1)
    torch_out = torch.matmul(attn_weights, V)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.softmax_attention(Q, K, V)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(Q[:2, :4])
    print(K[:2, :4])
    print(V[:2, :4])
    print(out[:2, :4])
    print(torch_out[:2, :4])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def conv2D():
    # 测试
    input_rows, input_cols = 1024, 1000
    kernel_rows, kernel_cols = 3, 5
    x = torch.randn(input_rows * input_cols, device="cuda", dtype=torch.float32)
    kernel = torch.randn(kernel_rows * kernel_cols, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.conv2d(
        x.view(1, 1, input_rows, input_cols), 
        kernel.view(1, 1, kernel_rows, kernel_cols),
        stride=1, padding="valid")
    ouput_rows = input_rows - kernel_rows + 1
    ouput_cols = input_cols - kernel_cols + 1
    torch_out = torch_out.view(ouput_rows, ouput_cols)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.conv2D(x, kernel, input_rows, input_cols, kernel_rows, kernel_cols)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")
    out = out.view(ouput_rows, ouput_cols)

    # 打印结果
    print(x.view(input_rows, input_cols)[:2, :4])
    print(kernel.view(kernel_rows, kernel_cols))
    print(out[:2, :4])
    print(torch_out[:2, :4])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def histogram():
    # 测试
    N = 1000000
    num_bins = 300
    x = torch.randint(0, num_bins, (N,), device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
    torch_out = torch.histc(x, bins=num_bins, min=0, max=num_bins-1)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.histogram(x, num_bins)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def prefxix_sum():
    # 测试
    x = torch.randn(1024 * 1024 * 1 + 23, device="cuda", dtype=torch.float32)
    
    # 用 PyTorch 实现
    torch_out = torch.cumsum(x, dim=0)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.prefix_sum(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[-10:])
    print(out[-10:])
    print(torch_out[-10:])

    # 检查结果
    assert torch.allclose(out, torch_out, rtol=1e-2, atol=1e-1)
    print("All tests passed!")


def dot_product():
    # 测试
    x = torch.randn(1024 * 1024 * 5 + 23, device="cuda", dtype=torch.float32)
    y = torch.randn(1024 * 1024 * 5 + 23, device="cuda", dtype=torch.float32)
    
    # 用 PyTorch 实现
    torch_out = torch.dot(x, y)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.dot_product(x, y)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")
    
    # 打印结果
    print(x[:10])
    print(y[:10])
    print(out)
    print(torch_out)

    # 检查结果
    assert torch.allclose(out, torch_out, rtol=1e-2, atol=1e-1)
    print("All tests passed!")


def sparse_matrix_vector_multiplication():
    # 测试
    M = 1029
    N = 2053
    A = torch.randn(M * N, device="cuda", dtype=torch.float32)
    x = torch.randn(N, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.matmul(A.view(M, N), x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.sparse_matrix_vector_multiplication(A, x, M, N, 42);
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(A.view(M, N)[:10, :10])
    print(x[:10])
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    assert torch.allclose(out, torch_out, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def gemm():
    # 测试
    M = 3303
    N = 6556
    K = 1922
    alpha = 0.5
    beta = 0.1
    A = torch.randn(M * K, device="cuda", dtype=torch.float16)
    B = torch.randn(K * N, device="cuda", dtype=torch.float16)
    C = torch.randn(M * N, device="cuda", dtype=torch.float16)

    # 用 PyTorch 实现
    torch_out = torch.matmul(alpha * A.view(M, K), B.view(K, N)) + beta * C.view(M, N)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.gemm(A, B, C, M, N, K, alpha, beta)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(A.view(M, K))
    print(B.view(K, N))
    print(C.view(M, N))
    print(out.view(M, N)[-4:, -4:])
    print(torch_out.view(M, N)[-4:, -4:])

    # 检查结果
    assert torch.allclose(out.view(M, N), torch_out.view(M, N), rtol=1e-1, atol=1e-1)
    print("All tests passed!")


def avg_categorical_cross_entropy_loss():
    # 测试
    N = 9852
    C = 852
    x = torch.randn(N, C, device="cuda", dtype=torch.float32)
    y = torch.randint(0, C, (N,), device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
     # Step 1: Compute log(sum(exp(z_jk))) for each sample j (numerically stable)
    # logsumexp(dim=1) computes log(sum(exp(Z[j,:]))) for all j (avoids overflow)
    log_sum_exp = torch.logsumexp(x, dim=1)  # Shape: (N,)
    # Step 2: Extract z_jy_j (logit of the true class for each sample)
    # Use gather to select Z[j, true_labels[j]] for all j
    z_true_class = x.gather(dim=1, index=y.unsqueeze(1)).squeeze(1)  # Shape: (N,)
    # Step 3: Compute per-sample loss and average over the batch
    per_sample_loss = log_sum_exp - z_true_class  # Shape: (N,)
    torch_out = per_sample_loss.mean()  # Scalar

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.avg_categorical_cross_entropy_loss(x.view(-1), y, N, C)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")
    
    # 打印结果
    print(x[:10, :10])
    print(y[:10])
    print(out)
    print(torch_out)

    # 检查结果
    assert torch.allclose(out, torch_out, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def mse_vec():
    # 测试
    N = 1024*1024*10 + 23
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    y = torch.randn(N, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.mse_loss(x, y, reduction="mean")

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.mse_vec(x, y)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(y[:10])
    print(out)
    print(torch_out)

    # 检查结果
    assert torch.allclose(out, torch_out, rtol=1e-3, atol=1e-3)
    print("All tests passed!")




if __name__ == "__main__":
    # vector_addition()
    # matrix_multiplication()
    # conv1D()
    # reverse_array()
    # transpose()
    # color_inversion()
    # relu()
    # leaky_relu()
    # rainbow_table()
    # matrix_copy()
    # count_array_elements()
    # count_array_elements_2D()
    # silu()
    # swiglu()
    # reduce_sum()
    # softmax_1D()
    # softmax_attention()
    # conv2D()
    # histogram()
    # prefxix_sum()
    # dot_product()
    # sparse_matrix_vector_multiplication()
    # gemm()
    # avg_categorical_cross_entropy_loss()
    mse_vec()