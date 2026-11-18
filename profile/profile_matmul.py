#coding=utf-8

import os
import sys

# 获取当前脚本所在的绝对路径 (即 profile 目录)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (即 current_dir 的父目录)
root_dir = os.path.dirname(current_dir)

# 将根目录添加到 sys.path 的最前面
# 这样 Python 就会优先在根目录下查找 leetgpu 模块
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


import torch
import time
import leetgpu


def warmup():
    print("Warming up...")
    M = 5120
    N = 4096
    K = 4096
    a = torch.rand(M, K, device="cuda", dtype=torch.float32) 
    b = torch.rand(K, N, device="cuda", dtype=torch.float32)
    for _ in range(10):
        _ = leetgpu.matmul(a, b, False)
    torch.cuda.synchronize()

def profile():
    print("Profiling...")
    M = 5120
    N = 4096
    K = 4096
    a = torch.rand(M, K, device="cuda", dtype=torch.float32) 
    b = torch.rand(K, N, device="cuda", dtype=torch.float32)
    
    out = leetgpu.matmul(a, b, False)
    
    torch.cuda.synchronize()
    print(">>> Done.")


if __name__ == "__main__":
    warmup()
    profile()


"""
# --set full: 收集所有可用的指标（最详细，但运行速度最慢，可能重放多次 kernel）
# -o report_linear_recurrence: 输出文件名为 report_linear_recurrence.ncu-rep
# --target-processes all: 确保捕获所有子进程（虽然 python 通常是单进程，但这是一个好习惯）
# --launch-timeout 60000: 设置启动超时时间（毫秒），防止因 warmup 时间长而超时

ncu --set full -o profile/report_matmul python3 profile/profile_matmul.py
"""