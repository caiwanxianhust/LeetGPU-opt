#include "utils.cuh"


#include <cuda.h>
#include <cfloat>
#include <torch/extension.h>

/*
Implement the k-means clustering algorithm for 2D points. Given arrays of x and y coordinates for data points, initial centroids, and other parameters, assign each point to the nearest centroid and update the centroids iteratively. The final centroids and labels should be stored in the output arrays.

Implementation Requirements
External libraries are not permitted
The solve function signature must remain unchanged
The final result must be stored in labels, final_centroid_x, and final_centroid_y
Example 1:
Input:
sample_size = 4, k = 2, max_iterations = 10
data_x = [1.0, 2.0, 8.0, 9.0]
data_y = [1.0, 2.0, 8.0, 9.0]
initial_centroid_x = [1.0, 8.0]
initial_centroid_y = [1.0, 8.0]
Output: (see reference implementation for expected output)
Constraints
1 ≤ sample_size ≤ 1000000
1 ≤ k ≤ 1000
All arrays are float32 except labels, which is int32
*/

namespace {
    constexpr int BLOCK_SIZE = 256;
}

static __host__ __device__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T>
__global__ void init_by_value_kernel(T* vec, int size, T value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        vec[tid] = value;
    }
}

/**
 * @brief    Calculate distance between each point and each centroid
 * @param    data_x, data_y, centroid_x, centroid_y are device pointers
 * @param    sample_size is the number of data points
 * @param    k is the number of centroids
 * @param    distance is device pointer
 * @return   void
 */
__global__ void calculate_distance_kernel(const float* data_x, const float* data_y, const float* centroid_x,
                                          const float* centroid_y, int* sample_labels, 
                                          int* cluster_size,
                                          int sample_size, int k) 
{
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    float min_distance = FLT_MAX;
    float dx, dy, distance;
    int nearest_centroid_id = -1;
    if (sample_id < sample_size) {
        for (int centroid_id = 0; centroid_id < k; centroid_id++) {
            dx = data_x[sample_id] - centroid_x[centroid_id];
            dy = data_y[sample_id] - centroid_y[centroid_id];
            distance = dx * dx + dy * dy;
            nearest_centroid_id = distance < min_distance ? centroid_id : nearest_centroid_id;
            min_distance = min(distance, min_distance);
        }

        sample_labels[sample_id] = nearest_centroid_id;
        atomicAdd(&cluster_size[nearest_centroid_id], 1);
    }
}

/**
 * @brief    Initialize centroid coordinates
 * @param    centroid_x, centroid_y, cluster_size are device pointers
 * @param    centroid_x, centroid_y size is k
 * @param    k is the number of centroids
 * @return   void
 */
__global__ void init_centroid_kernel(const int* cluster_size, float* centroid_x, float* centroid_y, int k) 
{
    int centroid_id = blockIdx.x * blockDim.x + threadIdx.x;
    int cluster_size_i = cluster_size[centroid_id];
    // 初始化中心点坐标
    // 如果当前中心点没有样本，则保留原值
    if (centroid_id < k && cluster_size_i > 0) {
        centroid_x[centroid_id] = 0;
        centroid_y[centroid_id] = 0;
    }
}

/**
 * @brief    Update centroid coordinates
 * @param    data_x, data_y, labels, centroid_x, centroid_y, cluster_size are device pointers
 * @param    data_x, data_y, labels size is sample_size
 * @param    centroid_x, centroid_y, cluster_size size is k
 * @param    sample_size is the number of data points
 * @param    k is the number of centroids
 * @return   void
 */
__global__ void update_centroid_kernel(const float* data_x, const float* data_y, const int* sample_labels,
    float* centroid_x, float* centroid_y, int* cluster_size, int sample_size, int k) 
{ 
    int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id < sample_size) {
        int centroid_id = sample_labels[sample_id];
        int cluster_size_i = cluster_size[centroid_id];
        atomicAdd(&centroid_x[centroid_id], data_x[sample_id] / (float)cluster_size_i);
        atomicAdd(&centroid_y[centroid_id], data_y[sample_id] / (float)cluster_size_i);
    }
}


/**
 * @brief    Fit one step of k-means clustering
 * @param    data_x, data_y, labels, centroid_x, centroid_y, cluster_size are device pointers
 * @param    data_x, data_y, labels size is sample_size
 * @param    centroid_x, centroid_y, cluster_size size is k
 * @param    sample_size is the number of data points
 * @param    k is the number of centroids
 * @return   void
 */
void fit_step(const float* data_x, const float* data_y, int* labels, float* centroid_x, float* centroid_y, 
    int* cluster_size, int sample_size, int k) 
{
    init_by_value_kernel<int><<<div_up(k, BLOCK_SIZE), BLOCK_SIZE>>>(cluster_size, k, 0);

    // 找到每个样本距离最近的中心点
    calculate_distance_kernel<<<div_up(sample_size, BLOCK_SIZE), BLOCK_SIZE>>>(
        data_x, data_y, centroid_x, centroid_y, labels, cluster_size,
        sample_size, k);
    
    // 初始化中心点坐标
    init_centroid_kernel<<<div_up(k, BLOCK_SIZE), BLOCK_SIZE>>>(
        cluster_size, centroid_x, centroid_y, k);

    // 更新中心点的坐标
    update_centroid_kernel<<<div_up(sample_size, BLOCK_SIZE), BLOCK_SIZE>>>(
        data_x, data_y, labels, centroid_x, centroid_y, cluster_size,
        sample_size, k);
}



/**
 * @brief    K-means clustering algorithm for 2D points
 * @param    data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
 * final_centroid_x, final_centroid_y are device pointers
 * @param    sample_size is the number of data points
 * @param    k is the number of clusters
 * @param    max_iterations is the maximum number of iterations
 * @return   void
 */
// data_x, data_y, labels, initial_centroid_x, initial_centroid_y,
// final_centroid_x, final_centroid_y are device pointers
void kmeans_clustering_cuda(const float* data_x, const float* data_y, int* labels,
                      float* initial_centroid_x, float* initial_centroid_y, float* final_centroid_x,
                      float* final_centroid_y, int sample_size, int k, int max_iterations) 
{
    // Allocate memory on device
    char* d_buf;
    // sample_labels[sample_size,], cluster_size[k,], centroid_x[k,], centroid_y[k,]
    size_t buf_size = k * sizeof(int);
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_buf, buf_size));
    int* cluster_size = (int*)(d_buf);
 
    for (int i = 0; i < max_iterations; i++) {
        fit_step(data_x, data_y, labels, initial_centroid_x, initial_centroid_y, cluster_size, sample_size, k);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(final_centroid_x, initial_centroid_x, k * sizeof(float),
                                cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(final_centroid_y, initial_centroid_y, k * sizeof(float),
                                cudaMemcpyDeviceToDevice));
    CHECK_CUDA_ERROR(cudaFree(d_buf));
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> kmeans_clustering(
    const torch::Tensor& data_x,
    const torch::Tensor& data_y,
    const torch::Tensor& init_centroids_x,
    const torch::Tensor& init_centroids_y,
    const int max_iterations
) { 
    TORCH_CHECK(data_x.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(data_y.ndimension() == 1, "Input must be 1D");
    TORCH_CHECK(init_centroids_x.ndimension() == 1, "init_centroids_x must be 1D");
    TORCH_CHECK(init_centroids_y.ndimension() == 1, "init_centroids_y must be 1D");
    TORCH_CHECK(data_x.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(data_y.device().is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(init_centroids_y.device().is_cuda(), "init_centroids_y must be on CUDA");
    TORCH_CHECK(init_centroids_x.device().is_cuda(), "init_centroids_x must be on CUDA");
    TORCH_CHECK(data_x.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(data_y.dtype() == torch::kFloat32, "Input must be of type float32");
    TORCH_CHECK(init_centroids_x.dtype() == torch::kFloat32, "init_centroids_x must be of type float32");
    TORCH_CHECK(init_centroids_y.dtype() == torch::kFloat32, "init_centroids_y must be of type float32");

    int sample_size = data_x.size(0);
    int k = init_centroids_x.size(0);
    TORCH_CHECK(sample_size > k, "Sample size must be greater than number of clusters");
    TORCH_CHECK(data_y.size(0) == sample_size, "Input must have the same number of rows as init_centroids_y");
    TORCH_CHECK(init_centroids_y.size(0) == k, "init_centroids_y must have the same number of rows as init_centroids_x");

    torch::Tensor labels = torch::empty({sample_size}, data_x.options().dtype(torch::kInt32));
    torch::Tensor final_centroids_x = torch::empty({k}, data_x.options());
    torch::Tensor final_centroids_y = torch::empty({k}, data_x.options());
    kmeans_clustering_cuda(
        data_x.data_ptr<float>(), 
        data_y.data_ptr<float>(), 
        labels.data_ptr<int>(), 
        init_centroids_x.data_ptr<float>(),
        init_centroids_y.data_ptr<float>(),
        final_centroids_x.data_ptr<float>(),
        final_centroids_y.data_ptr<float>(),
        sample_size, k, max_iterations
    );
    return {labels, final_centroids_x, final_centroids_y};
}