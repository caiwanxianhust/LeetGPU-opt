import torch
import leetgpu
import time
import torch.nn.functional as F


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



def matrix_multiplication():
    # 测试FP32
    M = 3000
    N = 2997
    K = 4001
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(N, K, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    torch_out = torch.matmul(a, b.transpose(0, 1))

    # 用 LeetGPU 实现
    start_time = time.time()
    c = leetgpu.matmul(a, b, True)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("Matrix A:")
    print(a[:4, :4])
    print("Matrix B:")
    print(b[:4, :4])
    print("Matrix C:")
    print(c)
    print(torch_out)

    # 检查结果
    compare_tensors(c, torch_out, atol=1e-2, rtol=1e-2, max_print=10)


def vector_addition():
    # 测试FP32
    N = 1000000
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)

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
    M = 8000
    N = 7559
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)

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
    M = 4006
    N = 3997
    x = torch.randint(0, 256, (M, N, 4), device="cuda", dtype=torch.uint8)

    # 用 PyTorch 实现
    torch_out = torch.empty_like(x, device="cuda", dtype=torch.uint8)
    torch_out[:, :, 0] = 255 - x[:, :, 0]
    torch_out[:, :, 1] = 255 - x[:, :, 1]
    torch_out[:, :, 2] = 255 - x[:, :, 2]
    torch_out[:, :, 3] = x[:, :, 3]
    # print("x: ")
    # print(x[:4, :4, :4])
    # print("pytorch out: ")
    # print(torch_out[:4, :4, :4])

    inp = x.flatten()
    print("input: ")
    print(inp[:16])
    
    # 用 LeetGPU 实现
    start_time = time.time()
    leetgpu.color_invert(inp, N, M)
    elapse_time = time.time() - start_time
    print(f"CUDA Time taken: {elapse_time * 1000} ms")

    # 打印结果
    print("cuda out: ")
    print(inp[:16])
    print("pytorch out: ")
    print(torch_out.flatten()[:16])

    # 检查结果
    compare_tensors(inp, torch_out.flatten(), atol=1e-3, rtol=1e-3, max_print=10)


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
    M = 4096
    N = 4096
    K = 3999
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
    print(A.view(M, K)[-4:, -4:])
    print(B.view(K, N)[-4:, -4:])
    print(C.view(M, N)[-4:, -4:])
    print(out.view(M, N)[-4:, -4:])
    print(torch_out.view(M, N)[-4:, -4:])

    # 检查结果
    compare_tensors(out.view(M, N), torch_out.view(M, N), rtol=1e-1, atol=1e-1)


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
    compare_tensors(out, torch_out, atol=1e-1, rtol=1e-1, max_print=10)


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
    N = 997
    M = 569
    K = 285
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


def kmeans_clustering():
    # 测试
    # 数据点数量
    N = 2000
    # 聚类中心数量
    k = 50
    # 迭代次数
    max_iterations = 10
    # 数据点 x 坐标
    data_x = torch.randn(N, device="cuda", dtype=torch.float32)
    # 数据点 y 坐标
    data_y = torch.randn(N, device="cuda", dtype=torch.float32)
    # 聚类中心 x 坐标
    centers_x = torch.randn(k, device="cuda", dtype=torch.float32)
    # 聚类中心 y 坐标
    centers_y = torch.randn(k, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # 针对数据集中每个样本  计算它到 k 个聚类中心的距离并将其分到距离最小的聚类中心所对应的类中，构成聚类结果 C(t)；
    # 计算新的类中心。对聚类结果 C(t)，计算每个类中的样本均值，作为类的新的中心；
     # 复制初始中心点
    curr_centers_x = centers_x.clone()
    curr_centers_y = centers_y.clone()
    
    data_points = torch.stack([data_x, data_y], dim=1)  # (N, 2)
    centers = torch.stack([curr_centers_x, curr_centers_y], dim=1)  # (k, 2)

    for iteration in range(max_iterations):
        # 计算每个点到每个中心的距离
        # data_points: (N, 2) -> (N, 1, 2)
        # centers: (k, 2) -> (1, k, 2)
        # distances: (N, k)
        distances = (data_points.unsqueeze(1) - centers.unsqueeze(0)) ** 2
        distances = distances.sum(dim=2)  # (N, k)

        # 找到最近的聚类中心
        labels = torch.argmin(distances, dim=1)  # (N,)
        
        # 更新聚类中心
        new_centers_x = torch.zeros_like(curr_centers_x)
        new_centers_y = torch.zeros_like(curr_centers_y)
        counts = torch.zeros(k, device=data_x.device, dtype=torch.float32)
        
        # 累加每个类别的点的坐标
        for i in range(k):
            mask = (labels == i)
            counts[i] = mask.sum()
            if counts[i] > 0:
                new_centers_x[i] = data_x[mask].sum() / counts[i]
                new_centers_y[i] = data_y[mask].sum() / counts[i]
            else:
                # 如果某类没有点，则保留原中心
                new_centers_x[i] = curr_centers_x[i]
                new_centers_y[i] = curr_centers_y[i]
        
        curr_centers_x = new_centers_x
        curr_centers_y = new_centers_y
        centers = torch.stack([curr_centers_x, curr_centers_y], dim=1)  # (k, 2)

    # 返回最终的标签和中心点
    torch_labels = labels.to(torch.int32)
    torch_final_centers_x = curr_centers_x
    torch_final_centers_y = curr_centers_y

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_kmeans_clustering(data_x, data_y, init_centers_x, init_centers_y, max_iter):
        labels, final_centers_x, final_centers_y = leetgpu.kmeans_clustering(
            data_x, data_y, init_centers_x, init_centers_y, max_iter
        )
        return labels, final_centers_x, final_centers_y

    gpu_labels, gpu_final_centers_x, gpu_final_centers_y = leetgpu_kmeans_clustering(
        data_x, data_y, centers_x, centers_y, max_iterations
    )

    # 打印结果
    print("Data Points (first 10)")
    print(data_x[:10])
    print(data_y[:10])
    print("Initial Centers")
    print(centers_x)
    print(centers_y)
    print("Final Centers (PyTorch)")
    print(torch_final_centers_x)
    print(torch_final_centers_y)
    print("Labels (first 10)")
    print(torch_labels[:10])
    print("Final Centers (GPU)")
    print(gpu_final_centers_x)
    print(gpu_final_centers_y)
    print("Labels (GPU, first 10)")
    print(gpu_labels[:10])

    # 检查结果
    compare_tensors(gpu_labels, torch_labels, atol=1e-4, rtol=1e-4, max_print=10)
    compare_tensors(gpu_final_centers_x, torch_final_centers_x, atol=1e-4, rtol=1e-4, max_print=10)
    compare_tensors(gpu_final_centers_y, torch_final_centers_y, atol=1e-4, rtol=1e-4, max_print=10)


def causal_self_attention():
    # 测试
    M = 500
    dmodel = 128
    # All elements in Q, K, and V are sampled from [-100.0, 100.0]，即元素范围在-100到100之间
    Q = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0
    K = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0
    V = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0

    # 用 PyTorch 实现
    qk = torch.matmul(Q, K.transpose(0, 1))
    qk = qk / math.sqrt(dmodel)
    # 对角线之后的元素为负无穷 masked(a_{ij}) = a_{ij} if i >= j else -inf
    mask = torch.triu(torch.ones(M, M, device="cuda", dtype=torch.float32), diagonal=1) * -1e9
    qk += mask
    attn_score = torch.softmax(qk, dim=-1)
    torch_out = torch.matmul(attn_score, V)
    
    # 用 LeetGPU 实现
    @timeit
    def leetgpu_causal_self_attention(Q, K, V):
        out = leetgpu.causal_self_attention(Q, K, V)
        return out
    
    out = leetgpu_causal_self_attention(Q, K, V)

    # 打印结果
    print("Q")
    print(Q[:4, :4])
    print("K")
    print(K[:4, :4])
    print("V")
    print(V[:4, :4])
    print("Causal Self-Attention Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def linear_self_attention():
    # 测试
    # Q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    # K = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    # V = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    # Q = torch.tensor(Q, device="cuda", dtype=torch.float32)
    # K = torch.tensor(K, device="cuda", dtype=torch.float32)
    # V = torch.tensor(V, device="cuda", dtype=torch.float32)
    M = 10000
    dmodel = 128
    Q = torch.rand(M, dmodel, device="cuda", dtype=torch.float32)
    K = torch.rand(M, dmodel, device="cuda", dtype=torch.float32)
    V = torch.rand(M, dmodel, device="cuda", dtype=torch.float32)
 
    # 用 PyTorch 实现
    # x+1 if x > 0 else exp(x)
    def elu(x):
        return torch.where(x > 0, x + 1, torch.exp(x))

    elu_kt = elu(K.transpose(0, 1))
    # print("elu_kt")
    # print(elu_kt)
    kv = torch.matmul(elu_kt, V)
    # print("elu_kt @ V")
    # print(kv)
    elu_q = elu(Q)
    # print("elu_q")
    # print(elu_q)
    qkv = torch.matmul(elu_q, kv)
    elu_k = elu(K)
    kp_sum = elu_k.sum(dim=0).reshape(-1, 1)
    # print("kp_sum")
    # print(kp_sum)
    torch_out = qkv / (torch.matmul(elu_q, kp_sum))
    # print("elu_q @ kp_sum")
    # print(torch.matmul(elu_q, kp_sum))

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_linear_self_attention(Q, K, V):
        out = leetgpu.linear_self_attention(Q, K, V)
        return out
    
    out = leetgpu_linear_self_attention(Q, K, V)

    # 打印结果
    print("Q")
    print(Q)
    print("K")
    print(K)
    print("V")
    print(V)
    print("Linear Self-Attention Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def sliding_window_self_attention():
    # 测试
    # Q = [[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0], [10.0, -11.0, 12.0]]
    # K = [[2.0, -1.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0], [-10.0, 11.0, -12.0]]
    # V = [[1.0, 0.5, -0.5], [-1.0, 2.0, 3.0], [4.0, -2.0, 1.0], [0.0, 1.0, -1.0]]
    # Q = torch.tensor(Q, device="cuda", dtype=torch.float32)
    # K = torch.tensor(K, device="cuda", dtype=torch.float32)
    # V = torch.tensor(V, device="cuda", dtype=torch.float32)
    # window_size = 2
    # M, dmodel = Q.shape
    M = 5000
    dmodel = 128
    Q = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0
    K = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0
    V = torch.rand(M, dmodel, device="cuda", dtype=torch.float32) * 100.0 - 50.0
    window_size = 16

    # 用 PyTorch 实现
    torch_out = torch.zeros_like(Q)
    for i in range(M):
        start = max(0, i - window_size)
        end = min(M, i + window_size + 1)
        qk = torch.matmul(Q[i:i+1], K[start:end].transpose(0, 1)) / math.sqrt(dmodel)
        attn_score = torch.softmax(qk, dim=-1)
        torch_out[i:i+1] = torch.matmul(attn_score, V[start:end])

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_sliding_window_self_attention(Q, K, V, window_size):
        out = leetgpu.sliding_window_self_attention(Q, K, V, window_size)
        return out
    
    out = leetgpu_sliding_window_self_attention(Q, K, V, window_size)

    # 打印结果
    print("Q")
    print(Q)
    print("K")
    print(K)
    print("V")
    print(V)
    print("Sliding Window Self-Attention Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def weight_dequantization():
    # 测试
    M = 300
    N = 200
    tile_size = 32
    X = torch.rand(M, N, device="cuda", dtype=torch.float32)
    S = torch.rand(((M + tile_size - 1) // tile_size), ((N + tile_size - 1) // tile_size), device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix X of shape [M, N] containing quantized values and a scale matrix S of shape [ceil(M/T), ceil(N/T)], where T is the tile size.
    # Y_{ij} = X_{ij} * S_{floor(i/T), floor(j/T)}
    torch_out = torch.zeros_like(X)
    for i in range(M):
        for j in range(N):
            torch_out[i, j] = X[i, j] * S[i // tile_size, j // tile_size]



    # 用 LeetGPU 实现
    @timeit
    def leetgpu_weight_dequantization(X, S, tile_size):
        out = leetgpu.weight_dequantization(X, S, tile_size)
        return out
    
    out = leetgpu_weight_dequantization(X, S, tile_size)

    # 打印结果
    print("X")
    print(X)
    print("S")
    print(S)
    print("Weight Dequantization Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def jacobi_stencil_2D():
    # 测试
    M = 1000
    N = 1000
    X = torch.rand(M, N, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # Implement a GPU program that applies a 2D Jacobi stencil to a matrix X of shape [M, N]. The stencil is defined as follows:
    # Given a 2D grid of 32-bit floating point values, apply one iteration of the 5-point Jacobi stencil: each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom, left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged from the input to the output.
    torch_out = X.clone()
     # 使用张量切片一次性处理所有内部点
    # 上方邻居: X[0:-2, 1:-1]
    # 下方邻居: X[2:, 1:-1]
    # 左侧邻居: X[1:-1, 0:-2]
    # 右侧邻居: X[1:-1, 2:]
    # 计算内部区域的平均值
    torch_out[1:-1, 1:-1] = (X[0:-2, 1:-1] + X[2:, 1:-1] + X[1:-1, 0:-2] + X[1:-1, 2:]) / 4.0

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_jacobi_stencil_2D(X):
        out = leetgpu.jacobi_stencil_2D(X)
        return out
    
    out = leetgpu_jacobi_stencil_2D(X)

    # 打印结果
    print("X")
    print(X)
    print("Jacobi Stencil 2D Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def stream_compaction():
    """
    Given a 1D array A of N 32-bit floating point numbers, compact all positive elements (A[i] > 0) to the front of the output array out, preserving their original relative order. Fill any remaining positions with 0.0. Stream compaction is a fundamental GPU primitive used throughout rendering, sparse computation, and collision detection.
    Implementation Requirements
    Use only native GPU features (external libraries are not permitted)
    The solve function signature must remain unchanged
    The first k positions of out must contain the k elements of A where A[i] > 0, in their original order
    Positions k through N−1 of out must be 0.0
    Elements where A[i] = 0.0 are not selected
    Example
    Input:  A = [1.0, -2.0, 3.0, 0.0, -1.0, 4.0]
    Output: out = [1.0, 3.0, 4.0, 0.0, 0.0, 0.0]
    Constraints
    1 ≤ N ≤ 100,000,000
    −1000.0 ≤ A[i] ≤ 1000.0
    out is pre-allocated with N elements, initialised to 0.0
    Performance is measured with N = 50,000,000
    """
    # 测试
    M = 100000
    A = torch.rand(M, device="cuda", dtype=torch.float32) * 10.0 - 5.0

    # 用 PyTorch 实现
    # 创建布尔掩码，标记正数元素
    mask = A > 0
    # 提取所有正数元素
    positive_elements = A[mask]
    # 创建输出张量，初始化为零
    torch_out = torch.zeros_like(A)
    # 将正数元素复制到输出张量的前部
    torch_out[:positive_elements.size(0)] = positive_elements

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_stream_compaction(A):
        out = leetgpu.stream_compaction(A)
        return out
    
    out = leetgpu_stream_compaction(A)

    # 打印结果
    print("A")
    print(A)
    print("Stream Compaction Result")
    print(out)
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def moe_topk_gating():
    """
MoE Top-K Gating
Medium
Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape [M, E] where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in the topk_weights and topk_indices arrays
Example 1:
Input:
  logits = [[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]]
  M = 2, E = 4, k = 2

Output:
  topk_weights = [[0.7311, 0.2689],
                  [0.7311, 0.2689]]
  topk_indices = [[3, 2],
                  [0, 1]]

Explanation:
Row 0: Top-2 values are 3.0 and 4.0 at indices 2 and 3.
       Softmax([3.0, 4.0]) = [0.2689, 0.7311]
Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
       Softmax([4.0, 3.0]) = [0.7311, 0.2689]
Constraints
1 ≤ M ≤ 10,000 (number of tokens)
1 ≤ E ≤ 256 (number of experts)
1 ≤ k ≤ E (top-k selection, typically k=2)
All tensors are stored on GPU
Logits are 32-bit floats
Indices are 32-bit integers
Performance is measured with M = 1,024, k = 2
    """
    # 测试
    M = 200
    E = 200
    k = int(4)
    logits = torch.rand(M, E, device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # 创建一个二维张量，用于存储Top-K的索引和权重
    touch_topk_weights = torch.zeros(M, k, device="cuda", dtype=torch.float32)
    touch_topk_indices = torch.zeros(M, k, device="cuda", dtype=torch.int32)
    # 使用张量切片一次性处理所有行
    for i in range(M):
        # 获取当前行
        row = logits[i]
        # 取前k个最大值及其原始索引
        topk_vals, topk_orig_indices = torch.topk(row, k)  # 获取top-k的值和原始索引
        
        touch_topk_indices[i] = topk_orig_indices
        # 对这些值应用softmax
        touch_topk_weights[i] = F.softmax(topk_vals, dim=0)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_moe_topk_gating(logits, k):
        topk_weights, topk_indices = leetgpu.moe_topk_gating(logits, k)
        return topk_weights, topk_indices
    
    topk_weights, topk_indices = leetgpu_moe_topk_gating(logits, k)

    # 打印结果
    print("Logits")
    print(logits)
    print("Top-K Weights")
    print(topk_weights[:10])
    print(touch_topk_weights[:10])
    print("Top-K Indices")
    print(topk_indices[:10])
    print(touch_topk_indices[:10])

    # 检查结果
    compare_tensors(topk_weights, touch_topk_weights, atol=1e-3, rtol=1e-3, max_print=10)
    compare_tensors(topk_indices, touch_topk_indices, max_print=10)


def gpt2_transformer_block():
    # 测试
    seq_len = 128
    dmodel = 768
    num_heads = 12
    head_dim = dmodel // num_heads
    weights_size = dmodel * 2 + \
            dmodel * 3 * dmodel + 3 * dmodel + \
            dmodel * dmodel + dmodel + \
            dmodel * 2 + \
            dmodel * 3072 + 3072 + \
            3072 * 768 + dmodel

    
    x = torch.randn(seq_len, dmodel, device="cuda", dtype=torch.float32) 
    weights = torch.randn(weights_size, device="cuda", dtype=torch.float32)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_gpt2_transformer_block(x, weights):
        out = leetgpu.gpt2_transformer_block(x, weights)
        return out
    
    out = leetgpu_gpt2_transformer_block(x, weights)

    # 用 PyTorch 实现
    # layer_normalization x_norm = layer_norm(x)
    off_gamma1 = 0;
    off_beta1 = dmodel
    off_w_qkv = 2 * dmodel
    off_b_qkv = 2 * dmodel + dmodel * 3 * dmodel
    off_w_attn = 2 * dmodel + dmodel * 3 * dmodel + 3 * dmodel
    off_b_attn = 2 * dmodel + dmodel * 3 * dmodel + 3 * dmodel + dmodel * dmodel
    off_gamma2 = off_b_attn + dmodel
    off_beta2 = off_gamma2 + dmodel
    off_w_fc = off_beta2 + dmodel
    off_b_fc = off_w_fc + dmodel * 3072
    off_w_proj = off_b_fc + 3072
    off_b_proj = off_w_proj + 3072 * dmodel
    print("gamma_1", off_gamma1, "beta_1", off_beta1, "w_qkv", off_w_qkv, "b_qkv", off_b_qkv, 
          "w_attn", off_w_attn, "b_attn", off_b_attn, "gamma_2", off_gamma2, "beta_2", off_beta2, 
          "w_fc", off_w_fc, "b_fc", off_b_fc, "w_proj", off_w_proj, "b_proj", off_b_proj)

    gamma_1 = weights[off_gamma1:off_beta1].view(1, dmodel)
    beta_1 = weights[off_beta1:off_w_qkv].view(1, dmodel)
    w_qkv = weights[off_w_qkv:off_b_qkv].view(dmodel, 3 * dmodel)
    bias_qkv = weights[off_b_qkv:off_w_attn].view(1, 3 * dmodel)
    w_attn = weights[off_w_attn:off_b_attn].view(dmodel, dmodel)
    bias_attn = weights[off_b_attn:off_gamma2].view(1, dmodel)
    gamma_2 = weights[off_gamma2:off_beta2].view(1, dmodel)
    beta_2 = weights[off_beta2:off_w_fc].view(1, dmodel)
    w_fc = weights[off_w_fc:off_b_fc].view(dmodel, 3072)
    bias_fc = weights[off_b_fc:off_w_proj].view(1, 3072)
    w_proj = weights[off_w_proj:off_b_proj].view(3072, dmodel)
    bias_proj = weights[off_b_proj:].view(1, dmodel)

    def layer_norm(x, gamma, beta):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        x_norm = (x - x_mean) / torch.sqrt(x_var + 1e-5) 
        return gamma * x_norm + beta

    x_norm = layer_norm(x, gamma_1, beta_1)
    print(x_norm.shape, w_qkv.shape)
    # matmul and bias
    qkv = torch.matmul(x_norm, w_qkv) + bias_qkv
    # maulti_head_attention
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(seq_len, num_heads, head_dim)
    k = k.view(seq_len, num_heads, head_dim)
    v = v.view(seq_len, num_heads, head_dim)
    # transpose,from (seq_len, num_heads, head_dim) to (num_heads, seq_len, head_dim)
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    print("Q, K, V shapes after reshape and transpose")
    print(q.shape, k.shape, v.shape)
    qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_scores = qk.softmax(dim=-1)
    attn_out = torch.matmul(attn_scores, v)
    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, dmodel)
    attn_projected_res = torch.matmul(attn_out, w_attn) + bias_attn + x
    # layer_normalization
    x_norm2 = layer_norm(attn_projected_res, gamma_2, beta_2)
    # feed_forward
    fc_linear = torch.matmul(x_norm2, w_fc) + bias_fc
    fc_gelu = F.gelu(fc_linear)
    fc_out = torch.matmul(fc_gelu, w_proj) + bias_proj + attn_projected_res

    

    # 打印结果
    print("Input X")
    print(x[:4])
    print("Output X")
    print(out)
    print(fc_out)
    # print(qkv)  # Q

    # 检查结果
    compare_tensors(out, fc_out, atol=1e-2, rtol=1e-2, max_print=10)


def all_pairs_shortest_paths():
    # 测试
    N = 4096
    adj_matrix = torch.randint(0, 10, (N, N), device="cuda", dtype=torch.float32)

    # 用 PyTorch 实现
    # 用 Floyd-Warshall 算法计算所有节点对之间的最短路径
    """
    下面是Floyd-Warshall算法的Python代码实现：
def floyd_warshall(graph):
   # 初始化距离矩阵
   dist = [[float('inf') for _ in range(len(graph))] for _ in range(len(graph))]
   for i in range(len(graph)):
       dist[i][i] = 0
   for i in range(len(graph)):
       for j in range(len(graph)):
           if graph[i][j] != 0:
               dist[i][j] = graph[i][j]
   # 动态规划更新最短路径
   for k in range(len(graph)):
       for i in range(len(graph)):
           for j in range(len(graph)):
               if dist[i][k] + dist[k][j] < dist[i][j]:
                   dist[i][j] = dist[i][k] + dist[k][j]
   return dist
    """

    @timeit
    def floyd_warshall_pytorch(graph):
        """
        使用PyTorch实现Floyd-Warshall算法计算所有节点对之间的最短路径
        
        参数:
        graph: torch.Tensor, 形状为[N, N]的邻接矩阵，其中graph[i][j]表示从节点i到节点j的边权重
            如果没有直接连接，则值为float('inf')
        
        返回:
        torch.Tensor: 最短路径距离矩阵
        """
        # 将输入图复制到一个新的张量中
        dist = graph.clone()
        N = dist.shape[0]
        
        # Floyd-Warshall算法的核心循环
        # 对于每个中间节点k，检查是否通过k可以缩短任意两点间的距离
        for k in range(N):
            # 使用向量化操作来替代内部的两层循环
            # dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            dist = torch.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
            
        return dist

     # 将对角线设为0（自己到自己的距离为0）
    adj_matrix.fill_diagonal_(float('inf'))
    adj_matrix[adj_matrix == 0] = float('inf')
    adj_matrix.fill_diagonal_(0)

    # 用 Floyd-Warshall 算法计算所有节点对之间的最短路径
    torch_out = floyd_warshall_pytorch(adj_matrix)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_all_pairs_shortest_paths(graph):
        leetgpu_out = leetgpu.all_pairs_shortest_paths(graph)
        return leetgpu_out
    
    out = leetgpu_all_pairs_shortest_paths(adj_matrix)
    
    # 打印结果
    print("Adjacency Matrix")
    print(adj_matrix[:5, :5])
    print("LeetGPU Output")
    print(out[:5, :5])
    print("PyTorch Output")
    print(torch_out[:5, :5])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def gqa():
    """
Implement Grouped Query Attention (GQA), the attention mechanism used in modern large language models such as LLaMA-3, Mistral, and Gemma. GQA reduces the KV-cache memory footprint during inference by sharing key and value heads across groups of query heads. Given query tensor Q with num_q_heads heads and key/value tensors K, V each with num_kv_heads heads, compute scaled dot-product attention where every group of num_q_heads / num_kv_heads consecutive query heads attends to the same key and value head. All tensors use float32.

scale = 1 / sqrt(head_dim)
scores = Q @ K^T * scale
weights = softmax(scores)
output = weights @ V

Implementation Requirements
Implement the function solve(Q, K, V, output, num_q_heads, num_kv_heads, seq_len, head_dim).
Do not change the function signature or use external libraries beyond the standard GPU frameworks.
Write the result into the provided output buffer.
num_q_heads is always divisible by num_kv_heads.
Use scaled dot-product attention with scale factor 1 / sqrt(head_dim) and a softmax over the key dimension.

Constraints
1 ≤ num_kv_heads ≤ num_q_heads ≤ 64
num_q_heads is divisible by num_kv_heads
1 ≤ seq_len ≤ 4,096
8 ≤ head_dim ≤ 256; head_dim is a multiple of 8
All tensor values are float32
Performance is measured with num_q_heads = 32, num_kv_heads = 8, seq_len = 1,024, head_dim = 128
    """
    num_q_heads = 32
    num_kv_heads = 8
    seq_len = 1024
    head_dim = 128

    # Q = [[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
    #      [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
    #      [[-1.0, 0.0, 0.5, 0.0], [0.0, -1.0, 0.0, 0.5], [0.5, 0.0, -1.0, 0.0]],
    #      [[0.0, 0.5, 0.0, -1.0], [0.5, 0.0, 0.0, -1.0], [0.0, 0.0, 0.5, 0.5]]]
    # K = [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
    #      [[0.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]]]
    # V = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
    #      [[-1.0, -2.0, -3.0, -4.0], [2, 3, 4, 5], [6, 7, 8, 9]]]
    # Q = torch.tensor(Q, device="cuda", dtype=torch.float32)
    # K = torch.tensor(K, device="cuda", dtype=torch.float32)
    # V = torch.tensor(V, device="cuda", dtype=torch.float32)

    Q = torch.rand(num_q_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    K = torch.rand(num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
    V = torch.rand(num_kv_heads, seq_len, head_dim, device="cuda", dtype=torch.float32)

    # LeetGPU 实现
    @timeit
    def leetgpu_gqa(Q, K, V):
        out = leetgpu.gqa(Q, K, V)
        return out
    out = leetgpu_gqa(Q, K, V)

    # 用 PyTorch 实现
    @timeit
    def grouped_query_attention_optimized(Q, K, V):
        """
        优化版的GQA实现，不需要显式传入维度参数，从tensor中推断
        
        Args:
            Q: Query tensor, shape (num_q_heads, seq_len, head_dim)
            K: Key tensor, shape (num_kv_heads, seq_len, head_dim)
            V: Value tensor, shape (num_kv_heads, seq_len, head_dim)
        
        Returns:
            Output tensor, shape (num_q_heads, seq_len, head_dim)
        """
        num_q_heads, seq_len, head_dim = Q.shape
        num_kv_heads = K.shape[0]
        
        # 计算缩放因子
        scale = 1.0 / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
        
        # 计算每组查询头的数量
        heads_per_group = num_q_heads // num_kv_heads
        
        # 重塑Q以便批量处理同一组内的查询头
        Q_reshaped = Q.view(num_kv_heads, heads_per_group, seq_len, head_dim)
        
        # 扩展K和V以匹配查询头的形状
        K_expanded = K.unsqueeze(1).expand(-1, heads_per_group, -1, -1)  # (num_kv_heads, heads_per_group, seq_len, head_dim)
        V_expanded = V.unsqueeze(1).expand(-1, heads_per_group, -1, -1)  # (num_kv_heads, heads_per_group, seq_len, head_dim)
        
        # 计算注意力分数: (num_kv_heads, heads_per_group, seq_len, seq_len)
        scores = torch.matmul(Q_reshaped, K_expanded.transpose(-2, -1)) * scale
        
        # 应用softmax
        weights = F.softmax(scores, dim=-1)  # (num_kv_heads, heads_per_group, seq_len, seq_len)
        
        # 计算输出: (num_kv_heads, heads_per_group, seq_len, head_dim)
        output = torch.matmul(weights, V_expanded)
        
        # 重塑回原始形状 (num_q_heads, seq_len, head_dim)
        output = output.view(num_q_heads, seq_len, head_dim)
        
        return output

    torch_out = grouped_query_attention_optimized(Q, K, V)

    # 打印结果
    print("Q")
    print(Q[:5, :5, :5])
    print("K")
    print(K[:5, :5, :5])
    print("V")
    print(V[:5, :5, :5])
    print("LeetGPU Output")
    print(out[:5, :5, :5])
    print("PyTorch Output")
    print(torch_out[:5, :5, :5])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def linear_recurrence():
    """
    Given two matrices a and x, each of shape [B, L] (batch size × sequence length), compute the linear recurrence h of shape [B, L] defined by: h[b, 0] = x[b, 0] and h[b, t] = a[b, t] × h[b, t−1] + x[b, t] for t ≥ 1. All values are float32. This operation is the core computational primitive of State Space Models (SSMs) such as Mamba, S4, and H3.

Linear Recurrence: h[t] = a[t] · h[t-1] + x[t]

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The result must be stored in the output tensor h

Constraints
1 ≤ B ≤ 256 (batch size)
1 ≤ L ≤ 65,536 (sequence length)
All values in a and x are float32
Performance is measured with B = 64, L = 16,384
    """
    # 测试
    B = 500
    L = 128 * 64
    a = torch.rand(B, L, device="cuda", dtype=torch.float32) 
    x = torch.rand(B, L, device="cuda", dtype=torch.float32)

    # 用 leetGPU 实现
    @timeit
    def leetgpu_linear_recurrence(a, x):
        out = leetgpu.linear_recurrence(a, x)
        return out

    out = leetgpu_linear_recurrence(a, x)

    # 用 PyTorch 实现
    @timeit
    def pytorch_linear_recurrence(a, x):
        h = torch.zeros_like(x)
        h[:, 0] = x[:, 0]
        for t in range(1, L):
            h[:, t] = a[:, t] * h[:, t-1] + x[:, t]
        return h

    torch_out = pytorch_linear_recurrence(a, x)

    # 打印结果
    print("Batch Size:", B, "Sequence Length:", L)
    print("a")
    print(a[:5, :5])
    print("x")
    print(x[:5, :5])
    print("LeetGPU Output")
    print(out[:5, :5])
    print("PyTorch Output")
    print(torch_out[:5, :5])

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)


def lora_linear():
    """
    https://leetgpu.com/challenges/lora-linear
Implement a LoRA (Low-Rank Adaptation) linear layer forward pass. Given an input matrix x of shape batch × d_in, a base weight matrix W of shape d_out × d_in, a LoRA down-projection matrix A of shape rank × d_in, and a LoRA up-projection matrix B of shape d_out × rank, compute output = x × WT + lora_scale × (x × AT) × BT. All tensors are float32.

Implementation Requirements
Implement the solve function; do not change its signature.
Do not use external libraries beyond those provided.
Write the result into output.

Constraints
1 ≤ batch ≤ 1,024
1 ≤ d_in, d_out ≤ 8,192
1 ≤ rank ≤ 256; rank < min(d_in, d_out)
All tensors are float32 on GPU.
Performance is measured with batch = 256, d_in = 4,096, d_out = 4,096, rank = 64
    """
    # 测试
    batch = 259
    d_in = 2999
    d_out = 3777
    rank = 64
    lora_scale = 0.5
    x = torch.randn(batch, d_in, device="cuda", dtype=torch.float32)
    W = torch.randn(d_out, d_in, device="cuda", dtype=torch.float32)
    A = torch.randn(rank, d_in, device="cuda", dtype=torch.float32)
    B = torch.randn(d_out, rank, device="cuda", dtype=torch.float32)

    # 用 LeetGPU 实现
    @timeit
    def leetgpu_lora_linear(x, W, A, B, lora_scale):
        out = leetgpu.lora_linear(x, W, A, B, lora_scale)
        return out

    out = leetgpu_lora_linear(x, W, A, B, lora_scale)

    # 用 PyTorch 实现
    @timeit
    def pytorch_lora_linear(x, W, A, B, lora_scale):
        out = torch.matmul(x, W.T) + lora_scale * torch.matmul(torch.matmul(x, A.T), B.T)
        # out = torch.matmul(torch.matmul(x, A.T), B.T)
        return out

    torch_out = pytorch_lora_linear(x, W, A, B, lora_scale)

    # 打印结果
    print("Batch Size:", batch, "Input Dimension:", d_in, "Output Dimension:", d_out, "Rank:", rank)
    print("x")
    print(x[:5, :5])
    print("W")
    print(W[:5, :5])
    print("A")
    print(A[:5, :5])
    print("B")
    print(B[:5, :5])
    print("LeetGPU Output")
    print(out)
    print("PyTorch Output")
    print(torch_out)

    # 检查结果
    compare_tensors(out, torch_out, atol=1e-3, rtol=1e-3, max_print=10)



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
    # multi_agent_simulation()
    # kmeans_clustering()
    # causal_self_attention()
    # linear_self_attention()
    # sliding_window_self_attention()
    # weight_dequantization()
    # jacobi_stencil_2D()
    # stream_compaction()
    # moe_topk_gating()
    # gpt2_transformer_block()
    # all_pairs_shortest_paths()
    # gqa()
    # linear_recurrence()
    lora_linear()