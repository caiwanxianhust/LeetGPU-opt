#include "utils.cuh"

/*
Merge Sorted Arrays
Medium
Given two arrays A of M elements and B of N elements, each sorted in non-decreasing order and containing 32-bit floating point numbers, produce a single sorted output array C of M + N elements containing all values from A and B in non-decreasing order. The sequential merge algorithm processes elements one-by-one, but an efficient GPU implementation must determine each output element's origin independently — without a serial dependency chain.

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in array C
Example 1
Input:
A = [1.0, 3.0, 5.0, 7.0]  (M = 4)
B = [2.0, 4.0, 6.0, 8.0]  (N = 4)

Output:
C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
Example 2
Input:
A = [1.0, 2.0, 3.0]  (M = 3)
B = [4.0]             (N = 1)

Output:
C = [1.0, 2.0, 3.0, 4.0]
Constraints
1 ≤ M, N ≤ 10,000,000
All elements are 32-bit floating point numbers
Input arrays are sorted in non-decreasing order and may contain duplicates
Performance is measured with M = 10,000,000, N = 10,000,000
*/

#include <cfloat>

__global__ void mergePathKernel(const float* A, const float* B, float* C, int M, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= M + N) return;
    
    // 确定搜索范围
    int begin = max(0, tid - N);
    int end = min(tid, M);
    
    // 二分搜索找到分割点
    while (begin < end) {
        int mid = (begin + end) / 2;
        int posB = tid - mid - 1;
        
        // 判断是否需要向右搜索
        bool pred = (posB < 0 || (mid < M && A[mid] <= B[posB]));
        if (pred) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    
    int posA = begin;
    int posB = tid - begin;
    
    // 合并元素
    float a_val = (posA < M) ? A[posA] : FLT_MAX;
    float b_val = (posB < N) ? B[posB] : FLT_MAX;
    
    C[tid] = (a_val <= b_val) ? a_val : b_val;
}


// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int total = M + N;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    mergePathKernel<<<blocks, threads>>>(A, B, C, M, N);
}