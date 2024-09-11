#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

/*
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float* input, float* output) 
{

}
*/

// 矩阵点乘
__global__ void dotProductKernel(float* input1, float* input2, float* output, int N) {
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalRow < N && globalCol < N) {
        output[globalRow * N + globalCol] = input1[globalRow * N + globalCol] * input2[globalRow * N + globalCol];
    }
}

int main() {
    int N = 128;  // 矩阵的大小


    float *h_input1, *h_input2, *h_output;
    h_input1 = (float*)malloc(N * N * sizeof(float));
    h_input2 = (float*)malloc(N * N * sizeof(float));
    h_output = (float*)malloc(N * N * sizeof(float));

    // 随机初始化
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < N * N; ++i) {
        h_input1[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_input2[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *d_input1, *d_input2, *d_output;
    cudaMalloc(&d_input1, N * N * sizeof(float));
    cudaMalloc(&d_input2, N * N * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));

    cudaMemcpy(d_input1, h_input1, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2;
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, dotProductKernel, d_input1, d_input2, d_output, N);
    }

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 释放内存
    cudaFree(d_input1);
    cudaFree(d_input2);
    cudaFree(d_output);
    free(h_input1);
    free(h_input2);
    free(h_output);

    return 0;
}