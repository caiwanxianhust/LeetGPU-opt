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


static __host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}


namespace naive_floyd_warshall {

constexpr int TILE_SIZE = 32;


__global__ void floyd_warshall_kernel(float* dist, int N, int k) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float ik = dist[i * N + k];
        float kj = dist[k * N + j];
        dist[i * N + j] = min(dist[i * N + j], ik + kj);
    }
}


// dist, output are device pointers (i.e. pointers to memory on the GPU)
void all_pairs_shortest_paths_naive_cuda(const float* dist, float* output, int N) {
    printf("Using naive Floyd-Warshall\n");
    // Copy input distance matrix to output (since Floyd-Warshall updates in-place)
    CHECK_CUDA_ERROR(cudaMemcpy(output, dist, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE));
    
    for (int k = 0; k < N; ++k) {
        floyd_warshall_kernel<<<gridDim, blockDim>>>(output, N, k);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}


} // namespace naive_floyd_warshall


namespace shared_mem_floyd_warshall { 

constexpr int TILE_SIZE = 32;


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

void all_pairs_shortest_paths_shared_mem_cuda(const float* dist, float* output, int N) {
    printf("Using shared memory Floyd-Warshall\n");
    // Copy input distance matrix to output (since Floyd-Warshall updates in-place)
    CHECK_CUDA_ERROR(cudaMemcpy(output, dist, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE));
    
    for (int k = 0; k < N; ++k) {
        floyd_warshall_shared_kernel<<<gridDim, blockDim>>>(output, N, k);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}


}   // namespace shared_mem_floyd_warshall


namespace reBourne87_floyd_warshall {

constexpr int TILE = 32;

// 16x16 threads, each thread updates a 2x2 patch -> covers 32x32 tile
constexpr int TX = 16;
constexpr int TY = 16;


static __device__ __forceinline__ float fminf_fast(float a, float b) {
    return a < b ? a : b;
}

__global__ void fw_phase1(float* d, int N, int r) {
    __shared__ float piv[TILE][TILE];

    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int base = r * TILE;

    // load pivot tile (r, r) into shared (2x2 per thread)
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            piv[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }
    __syncthreads();

    // Floyd-Warshall within pivot tile
    for (int k = 0; k < TILE; ++k) {
        __syncthreads();
        float a0 = piv[ty * 2 + 0][k];
        float a1 = piv[ty * 2 + 1][k];

        float b0 = piv[k][tx * 2 + 0];
        float b1 = piv[k][tx * 2 + 1];

        float& c00 = piv[ty * 2 + 0][tx * 2 + 0];
        float& c01 = piv[ty * 2 + 0][tx * 2 + 1];
        float& c10 = piv[ty * 2 + 1][tx * 2 + 0];
        float& c11 = piv[ty * 2 + 1][tx * 2 + 1];

        c00 = fminf_fast(c00, a0 + b0);
        c01 = fminf_fast(c01, a0 + b1);
        c10 = fminf_fast(c10, a1 + b0);
        c11 = fminf_fast(c11, a1 + b1);
    }
    __syncthreads();

    // write back pivot tile
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base + (ty * 2 + dy);
        if (i >= N) continue;
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base + (tx * 2 + dx);
            if (j >= N) continue;
            d[i * N + j] = piv[ty * 2 + dy][tx * 2 + dx];
        }
    }
}

__global__ void fw_phase2_row(float* d, int N, int r) {
    __shared__ float piv[TILE][TILE];
    __shared__ float blk[TILE][TILE];

    int jb = (int)blockIdx.x;
    if (jb == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int base_i = r * TILE;
    const int base_j = jb * TILE;
    const int base = r * TILE;

    // load pivot tile (r,r) -> piv
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            piv[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }

    // load block (r, jb) -> blk
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            blk[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }
    __syncthreads();

    // update (r, jb): d[i,j] = min(d[i,j], d[i,k] + d[k,j])
    for (int k = 0; k < TILE; ++k) {
        __syncthreads();
        float a0 = piv[ty * 2 + 0][k];
        float a1 = piv[ty * 2 + 1][k];

        float b0 = blk[k][tx * 2 + 0];
        float b1 = blk[k][tx * 2 + 1];

        float& c00 = blk[ty * 2 + 0][tx * 2 + 0];
        float& c01 = blk[ty * 2 + 0][tx * 2 + 1];
        float& c10 = blk[ty * 2 + 1][tx * 2 + 0];
        float& c11 = blk[ty * 2 + 1][tx * 2 + 1];

        c00 = fminf_fast(c00, a0 + b0);
        c01 = fminf_fast(c01, a0 + b1);
        c10 = fminf_fast(c10, a1 + b0);
        c11 = fminf_fast(c11, a1 + b1);
    }
    __syncthreads();

    // write back block (r, jb)
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        if (i >= N) continue;
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            if (j >= N) continue;
            d[i * N + j] = blk[ty * 2 + dy][tx * 2 + dx];
        }
    }
}

__global__ void fw_phase2_col(float* d, int N, int r) {
    __shared__ float piv[TILE][TILE];
    __shared__ float blk[TILE][TILE];

    int ib = (int)blockIdx.x;
    if (ib == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int base_i = ib * TILE;
    const int base_j = r * TILE;
    const int base = r * TILE;

    // load pivot tile (r,r) -> piv
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            piv[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }

    // load block (ib, r) -> blk
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            blk[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }
    __syncthreads();

    // update (ib, r): d[i,j] = min(d[i,j], d[i,k] + d[k,j])
    for (int k = 0; k < TILE; ++k) {
        __syncthreads();
        float a0 = blk[ty * 2 + 0][k];
        float a1 = blk[ty * 2 + 1][k];

        float b0 = piv[k][tx * 2 + 0];
        float b1 = piv[k][tx * 2 + 1];

        float& c00 = blk[ty * 2 + 0][tx * 2 + 0];
        float& c01 = blk[ty * 2 + 0][tx * 2 + 1];
        float& c10 = blk[ty * 2 + 1][tx * 2 + 0];
        float& c11 = blk[ty * 2 + 1][tx * 2 + 1];

        c00 = fminf_fast(c00, a0 + b0);
        c01 = fminf_fast(c01, a0 + b1);
        c10 = fminf_fast(c10, a1 + b0);
        c11 = fminf_fast(c11, a1 + b1);
    }
    __syncthreads();

    // write back block (ib, r)
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        if (i >= N) continue;
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            if (j >= N) continue;
            d[i * N + j] = blk[ty * 2 + dy][tx * 2 + dx];
        }
    }
}

__global__ void fw_phase3(float* d, int N, int r) {
    __shared__ float ik[TILE][TILE];
    __shared__ float kj[TILE][TILE];
    __shared__ float ij[TILE][TILE];

    int jb = (int)blockIdx.x;
    int ib = (int)blockIdx.y;
    if (ib == r || jb == r) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int base_i = ib * TILE;
    const int base_j = jb * TILE;
    const int base_r = r * TILE;

    // load (ib, r) -> ik
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int kcol = base_r + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && kcol < N) v = d[i * N + kcol];
            ik[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }

    // load (r, jb) -> kj
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int krow = base_r + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            float v = INFINITY;
            if (krow < N && j < N) v = d[krow * N + j];
            kj[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }

    // load (ib, jb) -> ij
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            float v = INFINITY;
            if (i < N && j < N) v = d[i * N + j];
            ij[ty * 2 + dy][tx * 2 + dx] = v;
        }
    }

    __syncthreads();

    // update (ib, jb) using intermediates in block r
    for (int k = 0; k < TILE; ++k) {
        __syncthreads();
        float a0 = ik[ty * 2 + 0][k];
        float a1 = ik[ty * 2 + 1][k];

        float b0 = kj[k][tx * 2 + 0];
        float b1 = kj[k][tx * 2 + 1];

        float& c00 = ij[ty * 2 + 0][tx * 2 + 0];
        float& c01 = ij[ty * 2 + 0][tx * 2 + 1];
        float& c10 = ij[ty * 2 + 1][tx * 2 + 0];
        float& c11 = ij[ty * 2 + 1][tx * 2 + 1];

        c00 = fminf_fast(c00, a0 + b0);
        c01 = fminf_fast(c01, a0 + b1);
        c10 = fminf_fast(c10, a1 + b0);
        c11 = fminf_fast(c11, a1 + b1);
    }
    __syncthreads();

    // write back (ib, jb)
    #pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int i = base_i + (ty * 2 + dy);
        if (i >= N) continue;
        #pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int j = base_j + (tx * 2 + dx);
            if (j >= N) continue;
            d[i * N + j] = ij[ty * 2 + dy][tx * 2 + dx];
        }
    }
}

// dist, output are device pointers (i.e. pointers to memory on the GPU)
void all_pairs_shortest_paths_tile_cuda(const float* dist, float* output, int N) {
    printf("Using reBourne87 tiled Floyd-Warshall\n");
    if (N <= 0) return;

    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    // Initialize output = dist (device-to-device)
    cudaMemcpy(output, dist, bytes, cudaMemcpyDeviceToDevice);

    const int rounds = (N + TILE - 1) / TILE;
    dim3 threads(TX, TY);

    for (int r = 0; r < rounds; ++r) {
        // Phase 1: pivot block
        fw_phase1<<<dim3(1, 1), threads>>>(output, N, r);

        // Phase 2: pivot row and pivot column
        fw_phase2_row<<<dim3(rounds, 1), threads>>>(output, N, r);
        fw_phase2_col<<<dim3(rounds, 1), threads>>>(output, N, r);

        // Phase 3: remaining blocks
        fw_phase3<<<dim3(rounds, rounds), threads>>>(output, N, r);
    }
}


}  // namespace reBourne87_floyd_warshall    


namespace tiled_tsinghua_floyd_warshall {

constexpr int TILE_SIZE = 32;

/**
 * Floyd-Warshall phase 1
 * @param dist Distance matrix
 * @param N Number of vertices
 * @param p Current pivot block index
 */
__global__ void fw_phase1(float* dist, int N, int p) { 
    // Thread indices
    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..31

    // Each block handles one pivot tile (p, p)
    __shared__ float piv[TILE_SIZE][TILE_SIZE];

    // Load pivot tile (p, p) into shared memory
    int row = p * TILE_SIZE + ty;
    int col = p * TILE_SIZE + tx;
    piv[ty][tx] = (row < N && col < N) ? dist[row * N + col] : INFINITY;
    __syncthreads();

    // Update pivot tile (p, p) using Floyd-Warshall
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) { 
        float ik = piv[ty][k];
        float kj = piv[k][tx];
        piv[ty][tx] = min(piv[ty][tx], ik + kj);
        __syncthreads();
    }
    
    // Write back pivot tile (p, p)
    if (row < N && col < N) dist[row * N + col] = piv[ty][tx];
}

void launch_fw_phase1(float* dist, int N, int p) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // Only one block is needed for the pivot tile
    fw_phase1<<<1, blockDim>>>(dist, N, p);
}


/**
 * Floyd-Warshall phase 2
 * @param dist Distance matrix
 * @param N Number of vertices
 * @param p Current pivot block index
 */
__global__ void fw_phase2(float* dist, int N, int p) { 
    // Thread indices
    const int tx = threadIdx.x; // 0..31
    const int ty = threadIdx.y; // 0..31
    const int bx = blockIdx.x;
    if (bx == p) return;

    // Each block handles one pivot tile
    // center tile (p, p)
    __shared__ float piv[TILE_SIZE][TILE_SIZE];
    // row tile (p, b)
    __shared__ float pb[TILE_SIZE][TILE_SIZE];
    // column tile (b, p)
    __shared__ float bp[TILE_SIZE][TILE_SIZE];

    // load center tile (p, p) -> piv
    int row = p * TILE_SIZE + ty;
    int col = p * TILE_SIZE + tx;
    piv[ty][tx] = (row < N && col < N) ? dist[row * N + col] : INFINITY;

    // load row tile (p, b) -> pb
    int row_pb = p * TILE_SIZE + ty;
    int col_pb = bx * TILE_SIZE + tx;
    pb[ty][tx] = (row_pb < N && col_pb < N) ? dist[row_pb * N + col_pb] : INFINITY;

    // load column tile (b, p) -> bp
    int row_bp = bx * TILE_SIZE + ty;
    int col_bp = p * TILE_SIZE + tx;
    bp[ty][tx] = (row_bp < N && col_bp < N) ? dist[row_bp * N + col_bp] : INFINITY;
    __syncthreads();

    // update row tile (p, b) using pivot tile (p, p) and original row tile (p, b)
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) { 
        float ik = piv[ty][k];
        float kb = pb[k][tx];
        pb[ty][tx] = min(pb[ty][tx], ik + kb);
    }

    // update column tile (b, p) using pivot tile (p, p) and original column tile (b, p)
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) { 
        float ik = bp[ty][k];
        float kj = piv[k][tx];
        bp[ty][tx] = min(bp[ty][tx], ik + kj);
    }
    __syncthreads();

    // write back row tile (p, b) and column tile (b, p)
    if (row_pb < N && col_pb < N) dist[row_pb * N + col_pb] = pb[ty][tx];
    if (row_bp < N && col_bp < N) dist[row_bp * N + col_bp] = bp[ty][tx];
}

void launch_fw_phase2(float* dist, int N, int p) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // Each block handles one pivot tile
    fw_phase2<<<div_up(N, TILE_SIZE), blockDim>>>(dist, N, p);
}


__global__ void fw_phase3(float* dist, int N, int p) { 
    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    if (bx == p || by == p) return;

    // Each block handles one tile
    // current tile (b, b)
    __shared__ float piv[TILE_SIZE][TILE_SIZE];
    // row tile (p, b)
    __shared__ float pb[TILE_SIZE][TILE_SIZE];
    // column tile (b, p)
    __shared__ float bp[TILE_SIZE][TILE_SIZE];

    // load center tile (p, p) -> piv
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    piv[ty][tx] = (row < N && col < N) ? dist[row * N + col] : INFINITY;

    // load kj tile (p, b) -> pb
    int row_pb = p * TILE_SIZE + ty;
    int col_pb = bx * TILE_SIZE + tx;
    pb[ty][tx] = (row_pb < N && col_pb < N) ? dist[row_pb * N + col_pb] : INFINITY;

    // load ik tile (b, p) -> bp
    int row_bp = by * TILE_SIZE + ty;
    int col_bp = p * TILE_SIZE + tx;
    bp[ty][tx] = (row_bp < N && col_bp < N) ? dist[row_bp * N + col_bp] : INFINITY;
    __syncthreads();

    // update tile (b, b) using pivot tile (p, p), row tile (p, b), and column tile (b, p)
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        float ik = bp[ty][k];
        float kj = pb[k][tx];
        piv[ty][tx] = min(piv[ty][tx], ik + kj);
        __syncthreads();
    }

    // write back tile (b, b)
    if (row < N && col < N) dist[row * N + col] = piv[ty][tx];
}


void launch_fw_phase3(float* dist, int N, int p) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(div_up(N, TILE_SIZE), div_up(N, TILE_SIZE));
    // Each block handles one tile
    fw_phase3<<<gridDim, blockDim>>>(dist, N, p);
}


void all_pairs_shortest_paths_tile_cuda(const float* dist, float* output, int N) { 
    printf("Using tiled_tsinghua_floyd_warshall\n");
    // Copy input distance matrix to output (since Floyd-Warshall updates in-place)
    CHECK_CUDA_ERROR(cudaMemcpy(output, dist, N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int p = 0; p < div_up(N, TILE_SIZE); ++p) { 
        launch_fw_phase1(output, N, p);
        CHECK_CUDA_ERROR(cudaGetLastError());

        launch_fw_phase2(output, N, p);
        CHECK_CUDA_ERROR(cudaGetLastError());

        launch_fw_phase3(output, N, p);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
}

}   // namespace tiled_tsinghua_floyd_warshall




void all_pairs_shortest_paths_cuda(const float* dist, float* output, int N) {
    // naive_floyd_warshall::all_pairs_shortest_paths_naive_cuda(dist, output, N);
    // shared_mem_floyd_warshall::all_pairs_shortest_paths_shared_mem_cuda(dist, output, N);
    // reBourne87_floyd_warshall::all_pairs_shortest_paths_tile_cuda(dist, output, N);
    tiled_tsinghua_floyd_warshall::all_pairs_shortest_paths_tile_cuda(dist, output, N);
}