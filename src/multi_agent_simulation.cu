#include "utils.cuh"
#include "cuda_kernels.cuh"

#include <cuda.h>


/*
Implement a program for a multi-agent flocking simulation (boids). The input consists of:

An array agents containing N agents, where N is the total number of agents
Each agent occupies 4 consecutive 32-bit floating point numbers in the array: 
, where:
 represents the agent's position in 2D space
 represents the agent's velocity vector
The total array size is 4 * N floats, with agent 
's data stored at indices [4i, 4i+1, 4i+2, 4i+3]
Simulation Rules
For each agent 
, identify all neighbors 
 (where 
) within radius 
 using:
Compute average velocity of neighboring agents:
 
 
 
 
where 
 is the set of neighbors for agent 
Update velocity:
Update position:
Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the agents_next array
Example 1:
Input: N = 2
agents = [
  0.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 4.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]

Output:
agents_next = [
  1.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
  3.0, 3.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
]
Constraints
1 ≤ N ≤ 100,000
Each agent's position and velocity components are 32-bit floats
*/

__device__ __forceinline__ float cal_distance(float4 agent, float4 neighbor) {
    return (agent.x - neighbor.x) * (agent.x - neighbor.x) + (agent.y - neighbor.y) * (agent.y - neighbor.y);
}


__global__ void multi_agent_sim_kernel(const float* agents, float* agents_next, int N) {

    const float4* agents4_ptr = reinterpret_cast<const float4*>(agents);
    float4* agents_next4_ptr = reinterpret_cast<float4*>(agents_next);
    int num_neighbors = 0;
    float4 val = agents4_ptr[blockIdx.x];
    float4 tmp;
    float2 avg_velocity = {0.0f, 0.0f};
    
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (i == blockIdx.x) continue;
        tmp = agents4_ptr[i];
        int is_neighbor_result = cal_distance(val, tmp) < 25.0f ? 1 : 0;
        num_neighbors += is_neighbor_result;
        if (is_neighbor_result) {
            avg_velocity.x += tmp.z;
            avg_velocity.y += tmp.w;
        }
    }
    __syncthreads();

    num_neighbors = block_all_reduce_sum(num_neighbors);
    float avg_velocity_x = block_all_reduce_sum(avg_velocity.x);
    float avg_velocity_y = block_all_reduce_sum(avg_velocity.y);

    if (threadIdx.x == 0) {
        avg_velocity.x = num_neighbors == 0 ? val.z : avg_velocity_x / (float)num_neighbors;
        avg_velocity.y = num_neighbors == 0 ? val.w : avg_velocity_y / (float)num_neighbors;

        float2 new_velocity = {
            val.z + 0.05f * (avg_velocity.x - val.z), 
            val.w + 0.05f * (avg_velocity.y - val.w)
        };
        agents_next4_ptr[blockIdx.x] = make_float4(
            val.x + new_velocity.x,
            val.y + new_velocity.y,
            new_velocity.x,
            new_velocity.y
        );
    }
}

// agents, agents_next are device pointers
void multi_agent_sim_cuda(const float* agents, float* agents_next, int N) 
{
    multi_agent_sim_kernel<<<N, 256>>>(agents, agents_next, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}