#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// 矩阵点乘
__global__ void matrixDotProduct(int8_t *matrixA, int8_t *matrixB, int8_t *result, int numRows, int numCols) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < numRows * numCols) {
        int row = index / numCols;
        int col = index % numCols;
        result[index] = matrixA[index] * matrixB[index];
    }
}

void checkCudaErrors(cudaError_t err) {
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int numRows = 128;  
    int numCols = 128;  

    size_t matrixSize = numRows * numCols * sizeof(int8_t);

    // 在cpu上分配内存
    int8_t *h_matrixA, *h_matrixB, *h_result;
    h_matrixA = (int8_t *)malloc(matrixSize);
    h_matrixB = (int8_t *)malloc(matrixSize);
    h_result = (int8_t *)malloc(matrixSize);

    // 随机初始化矩阵数据
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (size_t i = 0; i < matrixSize; ++i) {
        h_matrixA[i] = static_cast<int8_t>(std::rand() % 10);
        h_matrixB[i] = static_cast<int8_t>(std::rand() % 10);
    }

    // 在gpu上分配内存
    int8_t *d_matrixA, *d_matrixB, *d_result;
    checkCudaErrors(cudaMalloc(&d_matrixA, matrixSize));
    checkCudaErrors(cudaMalloc(&d_matrixB, matrixSize));
    checkCudaErrors(cudaMalloc(&d_result, matrixSize));

    checkCudaErrors(cudaMemcpy(d_matrixA, h_matrixA, matrixSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_matrixB, h_matrixB, matrixSize, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // kernel launch！
    matrixDotProduct<<<numBlocks, threadsPerBlock>>>(d_matrixA, d_matrixB, d_result, numRows, numCols);

    checkCudaErrors(cudaGetLastError());

    // d2h
    checkCudaErrors(cudaMemcpy(h_result, d_result, matrixSize, cudaMemcpyDeviceToHost));

    std::cout << "Matrix Dot Product Result:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << static_cast<int>(h_result[i * numCols + j]) << " ";
        }
        std::cout << std::endl;
    }

    // 释放资源
    checkCudaErrors(cudaFree(d_matrixA));
    checkCudaErrors(cudaFree(d_matrixB));
    checkCudaErrors(cudaFree(d_result));
    free(h_matrixA);
    free(h_matrixB);
    free(h_result);

    return 0;
}