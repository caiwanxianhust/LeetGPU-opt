import torch
import leetgpu
import time
import torch.nn.functional as F


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
    x = torch.randint(-127, 127, (1024, 10000), device="cuda", dtype=torch.int32)
    K = 0

    # 用 pytorch 实现
    torch_out = (x == K).sum().item()

    # 用 leetgpu 实现
    start_time = time.time()
    out = leetgpu.count_equal_2D(x.flatten(), 1024, 10000, K)
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


def gaussian_blur():
    # 测试
    input_rows, input_cols = 1025, 1025
    kernel_rows, kernel_cols = 5, 5
    x = torch.randn(input_rows, input_cols, device="cuda", dtype=torch.float32)
    kernel = torch.randn(kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.nn.functional.conv2d(
        x.view(1, 1, input_rows, input_cols), 
        kernel.view(1, 1, kernel_rows, kernel_cols),
        stride=1, padding="same")
    ouput_rows = input_rows
    ouput_cols = input_cols
    torch_out = torch_out.view(ouput_rows, ouput_cols)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.gaussian_blur(x, kernel)
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


def topk_1D():
    # 测试
    N = 10000000
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    # x = torch.arange(N, device="cuda", dtype=torch.float32)
    k = 1000

    # 用 PyTorch 实现
    torch_out = torch.topk(x, k, largest=True, sorted=True)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.topk_1D(x, k)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(x[:10])
    print(out[:10])
    print(torch_out.values[:10])

    # 检查结果
    assert torch.allclose(out, torch_out.values, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def batched_matmul():
    # 测试
    BATCH = 40
    M = 517
    N = 253
    K = 130
    A = torch.randn(BATCH, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.bmm(A, B)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.batched_matmul(A, B)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(A[0, :4, :4])
    print(B[0, :4, :4])
    print(out[0, :4, :4])
    print(torch_out[0, :4, :4])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def int8_quantized_matmul():
    # 测试
    M = 5000
    N = 4000
    K = 2041
    A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)
    scale_A = 0.05
    scale_B = 0.1
    scale_C = 0.01
    zero_point_A = 0
    zero_point_B = 0
    zero_point_C = 0

    # 用 PyTorch 实现
    A_float = A.int() - zero_point_A
    B_float = B.int() - zero_point_B
    C_float = torch.matmul(A_float.float(), B_float.float())
    torch_out = (scale_A * scale_B / scale_C) * C_float
    torch_out = torch.clamp(torch_out.round() + zero_point_C, -128, 127).to(torch.int8)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.int8_quantized_matmul(
        A, B, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print(A[:4, :4])
    print(B[:4, :4])
    print(out[-8:, :8])
    print(torch_out[-8:, :8])

    # 检查结果
    assert torch.allclose(out, torch_out, atol=1e-3)
    print("All tests passed!")


def ordinary_least_squares():
    # 测试
    n_samples = 1024 * 128  + 23
    n_features = 965
    x = torch.randn(n_samples, n_features, device="cuda", dtype=torch.float32)
    y = torch.randn(n_samples, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_beta = torch.linalg.lstsq(x, y).solution

    # 用 LeetGPU 实现
    start_time = time.time()
    beta = leetgpu.ordinary_least_squares(x, y)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X, y")
    print(x[:10, :10])
    print(y[:10])
    print("beta")
    print(beta[:10])
    print(torch_beta[:10])


    # 检查结果
    assert torch.allclose(beta, torch_beta, rtol=1e-2, atol=1e-2)
    print("All tests passed!")


class LogisticRegression(torch.nn.Module):
    def __init__(self, n_features, weight_decay=0.01):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=weight_decay)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


def logistic_regression():
    # 测试
    n_samples = 1024 * 128 + 23
    n_features = 985
    x = torch.randn(n_samples, n_features, device="cuda", dtype=torch.float32)
    y = torch.randint(0, 2, (n_samples,), device="cuda", dtype=torch.int32).float()
    # X = [[2.0, 1.0], [1.0, 2.0], [3.0, 3.0], [1.5, 2.5], [-1.0, -2.0], [-2.0, -1.0], [-1.5, -2.5], [-3.0, -3.0]]
    # y = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    # x = torch.tensor(X, device="cuda", dtype=torch.float32)
    # y = torch.tensor(y, device="cuda", dtype=torch.float32)
    # n_samples = 8
    # n_features = 2

    # 用 PyTorch 实现
    # train model
    model = LogisticRegression(n_features).to(torch.device("cuda"))
    init_beta = model.linear.weight.data[0].clone().detach()
    print("init weights")
    print(init_beta[:10])
    for epoch in range(50):
        y_pred = model(x)
        loss = model.loss_fn(y_pred, y.view(-1, 1))
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    print(loss.item())
    
    torch_beta = model.linear.weight.data[0].clone().detach()   

    # 用 LeetGPU 实现
    start_time = time.time()
    beta = leetgpu.logistic_regression(x, y, init_beta)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X, y")
    print(x[:10, :10])
    print(y[:10])
    print("beta")
    print(beta[:10])
    print(torch_beta[:10])

    # 检查结果
    assert torch.allclose(beta, torch_beta, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


# 双调排序
def sorting():
    # 测试
    N = 1024 * 128 + 23
    x = torch.randn(N, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_sorted = torch.sort(x)[0]

    # 用 LeetGPU 实现
    y = x.clone()
    start_time = time.time()
    leetgpu.sorting(y)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:10])
    print("sorted")
    print(y[:10])
    print(torch_sorted[:10])

    # 检查结果
    assert torch.allclose(y, torch_sorted, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def monte_carlo_integration():
    # 测试
    n_samples = 1024 * 128 + 23
    a, b = 2.0, 5.0
    y_samples = torch.randn(n_samples, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_integral = torch.mean(y_samples) * (b - a)

    # 用 LeetGPU 实现
    start_time = time.time()
    integral = leetgpu.monte_carlo_integration(y_samples, a, b)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("y_samples")
    print(y_samples[:10])
    print("integral")
    print(integral)
    print(torch_integral)

    # 检查结果
    assert torch.allclose(integral, torch_integral, rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def radix_sort():
    # 测试
    N = 1024 * 128 + 23
    # 修改为 uint32 类型
    x = torch.randint(0, 2**32, (N,), device="cuda", dtype=torch.uint32)

    # 用 PyTorch 实现 (正确处理 uint32 类型)
    # 将 uint32 转换为 float 进行排序，然后使用排序后的索引
    indices = torch.argsort(x.float())
    # 使用 gather 方法来获取排序后的结果，避免直接索引 uint32 张量
    torch_sorted = torch.gather(x, 0, indices)

    # 用 LeetGPU 实现
    y = x.clone()
    start_time = time.time()
    out = leetgpu.radix_sort(y)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:10])
    print("sorted")
    print(out[:10])
    print(torch_sorted[:10])

    # 检查结果 (比较前需要确保两个数组具有相同的类型)
    assert torch.allclose(out.float(), torch_sorted.float(), rtol=1e-3, atol=1e-3)
    print("All tests passed!")


def compare_tensors(tensor1, tensor2, atol=1e-3, rtol=1e-5, max_print=10):
    """
    比较两个张量，并打印所有不匹配元素的信息
    
    Args:
        tensor1, tensor2: 要比较的张量
        atol: 绝对容差
        rtol: 相对容差
        max_print: 最多打印多少个不匹配项
    
    Returns:
        bool: 如果所有元素都匹配则返回True
    """
    # 检查形状是否一致
    if tensor1.shape != tensor2.shape:
        print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
        return False
        
    # 计算差异
    close = torch.isclose(tensor1, tensor2, atol=atol, rtol=rtol)
    
    if close.all():
        print("All tests passed!")
        return True
    else:
        # 找到不匹配的位置
        not_close = ~close
        mismatch_indices = torch.nonzero(not_close)
        
        print(f"Found {mismatch_indices.shape[0]} mismatched elements out of {tensor1.numel()} total elements")
        print("\nFirst {} mismatches:".format(min(max_print, mismatch_indices.shape[0])))
        print("Index\t\tComputed\tExpected\tDifference")
        print("-" * 60)
        
        for i, idx in enumerate(mismatch_indices[:max_print]):
            idx_tuple = tuple(idx.tolist())
            computed_val = tensor1[idx_tuple].item()
            expected_val = tensor2[idx_tuple].item()
            diff = abs(computed_val - expected_val)
            
            print(f"{str(idx_tuple):15} {computed_val:10.6f} \t{expected_val:10.6f} \t{diff:10.6f}")
            
        # 显示最大差异
        abs_diff = torch.abs(tensor1 - tensor2)
        max_diff_idx = torch.argmax(abs_diff)
        max_diff_val = abs_diff.flatten()[max_diff_idx].item()
        
        # 将扁平索引转换为多维索引
        max_diff_multi_idx = torch.unravel_index(max_diff_idx, tensor1.shape) if hasattr(torch, 'unravel_index') else \
                            tuple((max_diff_idx // tensor1.stride(dim)) % tensor1.size(dim) for dim in range(tensor1.dim()))
        
        print(f"\nMaximum difference: {max_diff_val:.6f}")
        print(f"At index: {max_diff_multi_idx}")
        print(f"Computed value: {tensor1[max_diff_multi_idx].item():.6f}")
        print(f"Expected value: {tensor2[max_diff_multi_idx].item():.6f}")
        
        return False


def matrix_power():
    # 测试
    N = 200
    x = torch.randn(N, N, device="cuda", dtype=torch.float32)
    k = 7

    # 用 PyTorch 实现
    torch_out = torch.matrix_power(x, k)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.matrix_power(x, k)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")
    out = out.view(N, N)

    # 打印结果
    print("X")
    print(x[:4, :4])
    print("Power k")
    print(k)
    print("Result")
    print(out[-4:, -4:])
    print(torch_out[-4:, -4:])

    # 检查结果
    # assert torch.allclose(out, torch_out, atol=1e-3)
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def nearest_neighbor():
    # 测试
    N = 1024 * 128 + 23
    x = torch.randn(N, 3, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现 - 改进版本以避免内存溢出
    # 不再使用 torch.cdist 来计算完整的距离矩阵，因为这会占用太多内存
    # 用 PyTorch 实现计算最近邻索引
    def compute_nearest_neighbor_indices(points):
        indices = torch.zeros(points.shape[0], device=points.device, dtype=torch.int32)
        for i in range(points.shape[0]):
            # 计算当前点与其他所有点的距离
            diff = points - points[i]
            squared_distances = torch.sum(diff * diff, dim=1)
            # 设置自身距离为无穷大以排除自己
            squared_distances[i] = float('inf')
            # 找到最小距离的索引
            indices[i] = torch.argmin(squared_distances)
        return indices

    torch_out = compute_nearest_neighbor_indices(x)

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.nearest_neighbor(x)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4])
    print("Nearest Neighbor")
    print(out[:10])
    print(torch_out[:10])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def batch_norm():
    # 测试
    N = 1024
    C = 900
    x = torch.randn(N, C, device="cuda", dtype=torch.float32)
    gamma = torch.randn(C, device="cuda", dtype=torch.float32)
    beta = torch.randn(C, device="cuda", dtype=torch.float32)
    eps = 1e-5

    # 用 PyTorch 实现
    variance, mean = torch.var_mean(x, dim=0)
    normalized = (x - mean.view(1, -1)) / torch.sqrt(variance.view(1, -1) + eps)
    torch_out = gamma * normalized + beta

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.batch_norm(x, gamma, beta, eps)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4, :4])
    print("Batch Norm Result")
    print(out[:4, :4])
    print(torch_out[:4, :4])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def maxpooling_2D():
    # 测试
    N = 32
    C = 3
    H = 128
    W = 128
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 2
    padding = 1

    # 用 PyTorch 实现
    torch_out = F.max_pool2d(x, kernel_size, stride, padding)

    # 用 LeetGPU 实现
    start_time = time.time()    
    out = leetgpu.max_pooling_2D(x, kernel_size, stride, padding)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[0, 0, :4, :4])
    print("Max Pooling Result")
    print(out[0, 0, :4, :4])
    print(torch_out[0, 0, :4, :4])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def count_array_elements_3D():
    # 测试
    N = 32
    M = 32
    K = 10000
    P = 0
    # x = [P if (i%10== 0) else -1 for i in range(N * M * K)]
    x = torch.randint(-200, 200, (N * M * K, ), device="cuda", dtype=torch.int32)
    # x = torch.tensor(x, device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
    # 统计 x 中值等于 P 的元素个数
    torch_out = torch.sum(x == P).view(1).int()

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.count_equal_3D(x, N, M, K, P)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x.view(N, M, K)[:4, :4, :4])
    print("Count Equal Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def subarray_sum():
    # 测试
    N = 1024000000
    S = 10
    E = 5000
    x = torch.randint(1, 10, (N,), device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
    torch_out = torch.sum(x[S:(E+1)]).view(1).int()

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.subarray_sum(x, S, E)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4])
    print("Subarray Sum")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def subarray_sum_2D():
    # 测试
    N = 1000
    M = 820
    S_ROW, S_COL = 100, 100
    E_ROW, E_COL = 700, 400
    x = torch.randint(1, 10, (N, M), device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
    torch_out = torch.sum(x[S_ROW:E_ROW+1, S_COL:E_COL+1]).view(1).int()

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.subarray_sum_2D(x, S_ROW, E_ROW, S_COL, E_COL)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4, :4])
    print("Subarray Sum")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def subarray_sum_3D():
    # 测试
    N = 30
    M = 20
    K = 16
    S_DEP, S_ROW, S_COL = 0, 0, 0
    E_DEP, E_ROW, E_COL = 29, 19, 15
    # x = torch.randint(1, 10, (N, M, K), device="cuda", dtype=torch.int32)
    x = torch.ones(N, M, K, device="cuda", dtype=torch.int32)


    # 用 PyTorch 实现
    torch_out = torch.sum(x[S_DEP:E_DEP+1, S_ROW:E_ROW+1, S_COL:E_COL+1]).view(1).int()

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.subarray_sum_3D(x, S_DEP, E_DEP, S_ROW, E_ROW, S_COL, E_COL)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4, :4, :4])
    print("Subarray Sum")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def rms_norm_1D():
    # 测试
    N = 1024
    x = [-71.76253509521484, -80.75570678710938, 13.834547996520996, -13.680577278137207, 54.231380462646484, -17.70029640197754, 90.30525970458984, 66.61432647705078] 
    # x = torch.randn(N, device="cuda", dtype=torch.float32)
    x = torch.tensor(x, device="cuda", dtype=torch.float32)
    eps = 1e-5
    gamma = 1.5
    beta = 0.0

    # 用 PyTorch 实现
    rms = torch.sqrt(torch.mean(x * x) + eps)
    torch_out = x / rms * gamma + beta

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.rms_norm_1D(x, eps, gamma, beta)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4])
    print("RMS Norm Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-5, rtol=1e-5, max_print=10)


def max_subarray_sum():
    # 测试
    N = 1024 * 128 + 23
    window_size = 128
    x = torch.randint(-100, 100, (N,), device="cuda", dtype=torch.int32)

    # 用 PyTorch 实现
    max_sum = torch.ones(1, device="cuda", dtype=torch.int32)
    for i in range(N - window_size + 1):
        current_sum = torch.sum(x[i:i + window_size])
        if max_sum is None or current_sum > max_sum:
            max_sum = current_sum
    torch_out = max_sum.view(1).int()

    # 用 LeetGPU 实现
    start_time = time.time()
    out = leetgpu.max_subarray_sum(x, window_size)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("X")
    print(x[:4])
    print("Max Subarray Sum")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


from functools import wraps

# 计时装饰器
def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapse_time = time.time() - start_time
        print(f"Time taken: {elapse_time * 1000} ms")
        return result
    return wrapper


import math


def attention_with_linear_bias():
    # 测试
    N = 1024
    M = 128
    d = 512
    alpha = 0.5
    q = torch.randn(M, d, device="cuda", dtype=torch.float32)
    k = torch.randn(N, d, device="cuda", dtype=torch.float32)
    v = torch.randn(N, d, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # attn_scores[i, j] = qk^T / sqrt(d) + alpha * (i - j)
    attn_scores = torch.matmul(q, k.transpose(0, 1)) / math.sqrt(d) 
    attn_scores += alpha * (torch.arange(M, device="cuda", dtype=torch.float32).view(M, 1) - torch.arange(N, device="cuda", dtype=torch.float32).view(1, N))
    torch_out = torch.softmax(attn_scores, dim=-1)
    torch_out = torch.matmul(torch_out, v)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_attention_with_linear_bias(q, k, v, alpha):
        attn_scores = leetgpu.attention_with_linear_bias(q, k, v, alpha)
        return attn_scores
    
    out = leetgpu_attention_with_linear_bias(q, k, v, alpha)

    # 打印结果
    print("Q")
    print(q[:4, :4])
    print("K")
    print(k[:4, :4])
    print("V")
    print(v[:4, :4])
    print("Attention Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def batched_matrix_mul_fp16():
    # 测试
    BATCH = 32
    N = 2000
    M = 199
    K = 498
    A = torch.randn(BATCH, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(BATCH, K, N, device="cuda", dtype=torch.float16)

    # 用 PyTorch 实现
    torch_out = torch.matmul(A, B)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_batched_matrix_mul_fp16(A, B):
        out = leetgpu.batched_matrix_mul_fp16(A, B)
        return out

    out = leetgpu_batched_matrix_mul_fp16(A, B)

    # 打印结果
    print("A")
    print(A[:4, :4, :4])
    print("B")
    print(B[:4, :4, :4])
    print("Matrix Multiplication Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def dot_product_fp16():
    # 测试
    N = 100000000
    x = torch.randn(N, device="cuda", dtype=torch.float16)
    y = torch.randn(N, device="cuda", dtype=torch.float16)

    # 用 PyTorch 实现
    torch_out = torch.dot(x.float(), y.float())
    # 将 torch_out 转换为 float16
    torch_out = torch_out.to(torch.float16)



    # 用 LeetGPU 实现
    @timeit
    def leetgpu_dot_product_fp16(x, y):
        out = leetgpu.dot_product_fp16(x, y)
        return out
    
    out = leetgpu_dot_product_fp16(x, y)

    # 打印结果
    print("X")
    print(x[:4])
    print("Y")
    print(y[:4])
    print("Dot Product")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out.view(1), torch_out.view(1), atol=1e-2, rtol=1e-2, max_print=10)


def rope_embedding():
    # 测试
    M = 1024
    d = 512
    x = torch.randn(M, d, device="cuda", dtype=torch.float32)
    cos_table = torch.randn(M, d, device="cuda", dtype=torch.float32)
    sin_table = torch.randn(M, d, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # rope(x) = x * cos_table + rote_half(x) * sin_table
    # rote_half(x) = [-x_{d/2+1}, ..., -x_{d}, x_1, ..., x_{d/2-1}, x_{d/2}]
    torch_out = x * cos_table + torch.cat([-x[:, d // 2:], x[:, :d // 2]], dim=1) * sin_table

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_rope_embedding(x, cos_table, sin_table):
        out = leetgpu.rope_embedding(x, cos_table, sin_table)
        return out

    out = leetgpu_rope_embedding(x, cos_table, sin_table)

    # 打印结果
    print("X")
    print(x[:4, :4])
    print("Cos Table")
    print(cos_table[:4, :4])
    print("Sin Table")
    print(sin_table[:4, :4])
    print("Rope Embedding Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def conv3D():
    # 测试
    input_depth = 320
    input_height = 100
    input_width = 201
    kernel_depth = 3
    kernel_rows = 3
    kernel_cols = 3
    x = torch.randn(input_depth, input_height, input_width, device="cuda", dtype=torch.float32)
    kernel = torch.randn(kernel_depth, kernel_rows, kernel_cols, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # 无需 padding，stride=1
    output_depth = input_depth - kernel_depth + 1
    output_height = input_height - kernel_rows + 1
    output_width = input_width - kernel_cols + 1
    torch_out = F.conv3d(
        x.view(1, 1, input_depth, input_height, input_width),
        kernel.view(1, 1, kernel_depth, kernel_rows, kernel_cols),
        padding=0,
        stride=1,
    ).view(output_depth, output_height, output_width)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_conv3D(x, kernel):
        out = leetgpu.conv3D(x, kernel)
        return out
    
    out = leetgpu_conv3D(x, kernel)

    # 打印结果
    print("X")
    print(x[:4, :4, :4])
    print("Kernel")
    print(kernel[:4, :4, :4])
    print("Convolution Result")
    print(out[:4, :4, :4])
    print(torch_out[:4, :4, :4])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def mha():
    # 测试
    N = 1299                                                             
    d_model = 200
    num_heads = 8

    q = torch.randn(N, d_model, device="cuda", dtype=torch.float32)
    k = torch.randn(N, d_model, device="cuda", dtype=torch.float32)
    v = torch.randn(N, d_model, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    d_k = d_model // num_heads
    q_reshaped = q.view(N, num_heads, d_k).transpose(0, 1)  # (num_heads, N, d_k)
    k_reshaped = k.view(N, num_heads, d_k).transpose(0, 1)  # (num_heads, N, d_k)
    v_reshaped = v.view(N, num_heads, d_k).transpose(0, 1)  # (num_heads, N, d_k)
    torch_out = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / math.sqrt(d_k)  # (num_heads, N, N)
    torch_out = torch.softmax(torch_out, dim=-1)
    torch_out = torch.matmul(torch_out, v_reshaped)  # (num_heads, N, d_k)
    torch_out = torch_out.transpose(0, 1).contiguous().view(N, d_model)  # (N, d_model)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_mha(q, k, v, num_heads):
        out = leetgpu.mha(q, k, v, num_heads)
        return out

    out = leetgpu_mha(q, k, v, num_heads)

    # 打印结果
    print("Q")
    print(q[:4, :4])
    print("K")
    print(k[:4, :4])
    print("V")
    print(v[:4, :4])
    print("MHA Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-4, rtol=1e-4, max_print=10)


def multi_agent_simulation():
    # 测试
    N = 10000
    x = torch.randn(N, 4, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.zeros(N, 4, device="cuda", dtype=torch.float32)
    for i in range(N):
        num_neighbors = 0
        avg_velocity = torch.zeros(2, device="cuda", dtype=torch.float32)
        for j in range(N):
            if (i == j): 
                continue
            distance = x[i, :2] - x[j, :2]
            dist_squared = distance.pow(2).sum()
            if (dist_squared < 25.0):
                num_neighbors += 1
                avg_velocity += x[j, 2:]
        
        if (num_neighbors > 0):
            avg_velocity /= num_neighbors
        else:
            avg_velocity = x[i, 2:]

        torch_out[i, 2:] = x[i, 2:] + 0.05 * (avg_velocity - x[i, 2:])
        torch_out[i, :2] = x[i, :2] + torch_out[i, 2:]

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_multi_agent_simulation(x):
        out = leetgpu.multi_agent_sim(x)
        return out
    
    out = leetgpu_multi_agent_simulation(x)

    # 打印结果
    print("X")
    print(x[:4, :4])
    print("Multi-Agent Simulation Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-4, rtol=1e-4, max_print=10)




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
    # mse_vec()
    # gaussian_blur()
    # topk_1D()
    # batched_matmul()
    # int8_quantized_matmul()
    # ordinary_least_squares()
    # logistic_regression()
    # monte_carlo_integration()
    # sorting()
    # radix_sort()
    # matrix_power()
    # nearest_neighbor()
    # batch_norm()
    # maxpooling_2D()
    # count_array_elements_3D()
    # subarray_sum()
    # subarray_sum_2D()
    # subarray_sum_3D()
    # rms_norm_1D()
    # max_subarray_sum()
    # attention_with_linear_bias()
    # batched_matrix_mul_fp16()
    # dot_product_fp16()
    # rope_embedding()
    # conv3D()
    # mha()
    multi_agent_simulation()