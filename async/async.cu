#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>

void checkCudaErrors(cudaError_t err) {
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 计算矩阵平均值
__device__ void calculateAverage(float* data, int dim, float* dimAverages, int numElements, int totalDims) {
    float sum = 0.0f;
    for (int i = 0; i < numElements; ++i) {
        sum += data[dim * numElements + i];
    }
    dimAverages[dim] = sum / numElements;
}

// 对一个N维矩阵求平均值。先并行对每一维的矩阵调用calculateAverage求平均，等待所有维数据返回后，求总体平均。
__global__ void splitArriveWait(float *data, float *dimAverages, float *overallAverage, int numElements) {
    using barrier = cuda::barrier<cuda::thread_scope_block>;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); 
    }
    block.sync();

    for (int dim = 0; dim < 10; ++dim) {
        // 按维度求平均值
        barrier::arrival_token token = bar.arrive(); 
        calculateAverage(data, dim, dimAverages, numElements, 10);
        bar.wait(std::move(token)); // 等待
    }

    // 等待所有维度计算完成后，计算整体平均值
    float totalSum = 0.0f;
    for (int dim = 0; dim < 10; ++dim) {
        totalSum += dimAverages[dim];
    }
    *overallAverage = totalSum / 10;
}

int main() {
    int numDims = 10;
    int numElements = 256 * 256;  

    float *h_data, *h_dimAverages, *h_overallAverage;
    float *d_data, *d_dimAverages, *d_overallAverage;

    h_data = (float *)malloc(numDims * numElements * sizeof(float));
    h_dimAverages = (float *)malloc(numDims * sizeof(float));
    h_overallAverage = (float *)malloc(sizeof(float));

    // 随机初始化
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int dim = 0; dim < numDims; ++dim) {
        for (int i = 0; i < numElements; ++i) {
            h_data[dim * numElements + i] = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    checkCudaErrors(cudaMalloc(&d_data, numDims * numElements * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_dimAverages, numDims * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_overallAverage, sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_data, h_data, numDims * numElements * sizeof(float), cudaMemcpyHostToDevice));

    // kernel launch！
    splitArriveWait<<<1, 256>>>(d_data, d_dimAverages, d_overallAverage, numElements);

    checkCudaErrors(cudaMemcpy(h_dimAverages, d_dimAverages, numDims * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_overallAverage, d_overallAverage, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Overall Average: " << h_overallAverage[0] << std::endl;

    // 释放内存
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_dimAverages));
    checkCudaErrors(cudaFree(d_overallAverage));
    free(h_data);
    free(h_dimAverages);
    free(h_overallAverage);

    // 检查 CUDA error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaDeviceReset();

    return 0;
}