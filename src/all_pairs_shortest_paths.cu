#include "utils.cuh"

#include <cuda.h>


/*
Given a weighted directed graph of N vertices represented as an N × N distance matrix, compute the shortest path distance between every pair of vertices using the Floyd-Warshall algorithm. The matrix is stored as a flat array in row-major order: dist[i * N + j] is the weight of the directed edge from vertex i to vertex j. A value of +infinity means no direct edge exists. The diagonal is always zero. For each intermediate vertex k from 0 to N - 1 (in order), update all pairs:


Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in output
Example:
Input: N = 4
dist = [
  0,   5, inf,  10,   // row 0: edges from vertex 0
  inf, 0,   3, inf,   // row 1: edges from vertex 1
  inf, inf, 0,   1,   // row 2: edges from vertex 2
  inf, inf, inf, 0    // row 3: edges from vertex 3
]

Output:
output = [
  0,   5,   8,   9,   // shortest paths from vertex 0
  inf, 0,   3,   4,   // shortest paths from vertex 1
  inf, inf, 0,   1,   // shortest paths from vertex 2
  inf, inf, inf, 0    // shortest paths from vertex 3
]

Explanation:
- output[0][2] = 8   (path 0 -> 1 -> 2, cost 5 + 3 = 8)
- output[0][3] = 9   (path 0 -> 1 -> 2 -> 3, cost 5 + 3 + 1 = 9, beats direct 0 -> 3 = 10)
- output[1][3] = 4   (path 1 -> 2 -> 3, cost 3 + 1 = 4)
Constraints
1 ≤ N ≤ 4,096
Edge weights are finite float32 values or +infinity (no edge)
The input contains no negative cycles
The diagonal satisfies dist[i * N + i] = 0 for all i
dist and output are flat arrays of N × N floats in row-major order
Performance is measured with N = 2,048

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
*/

namespace {
    constexpr int TILE_SIZE = 32;
}

static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void floyd_warshall_kernel(float* dist, int N, int k) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float ik = dist[i * N + k];
        float kj = dist[k * N + j];
        dist[i * N + j] = min(dist[i * N + j], ik + kj);
    }
}


__global__ void floyd_warshall_shared_kernel(float* dist, int N, int k) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_ik[TILE_SIZE];
    __shared__ float s_kj[TILE_SIZE];

    if (threadIdx.x == 0 && i < N) {
        s_ik[threadIdx.y] = dist[i * N + k];
    }
    if (threadIdx.y == 0 && j < N) {
        s_kj[threadIdx.x] = dist[k * N + j];
    }
    __syncthreads();

    if (i < N && j < N) {
        dist[i * N + j] = min(dist[i * N + j], s_ik[threadIdx.y] + s_kj[threadIdx.x]);
    }
}




// dist, output are device pointers (i.e. pointers to memory on the GPU)
void all_pairs_shortest_paths_cuda(const float* dist, float* output, int N) {
    // Copy input distance matrix to output (since Floyd-Warshall updates in-place)
    CHECK_CUDA_ERROR(cudaMemcpy(output, dist, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE));
    
    for (int k = 0; k < N; ++k) {
        // floyd_warshall_shared_kernel<<<gridDim, blockDim>>>(output, N, k);
        floyd_warshall_kernel<<<gridDim, blockDim>>>(output, N, k);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}