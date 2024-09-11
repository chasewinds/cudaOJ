#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// 矩阵点乘，int8实现，可调度到INT8 Core上                                                 16            512           32
__global__ void matrixDotProduct(int8_t *matrixA, int8_t *matrixB, int8_t *result, int numRowsA, int numColsA, int numColsB) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;// 当前线程的行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程的列索引

    if (row < numRowsA && col < numColsB) {
        int8_t dotProduct = 0;  // 暂存一行乘一列加和结果
        for (int k = 0; k < numColsA; ++k) {  // 在一个执行线程中，计算一个行乘列的加和。k遍历公共维度
            dotProduct += matrixA[row * numColsA + k] * matrixB[k * numColsB + col];
        }
        result[row * numColsB + col] = dotProduct;
    }
}

void checkCudaErrors(cudaError_t err) {
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // 计算16 X 512 dot 512 X 32，返回 16 X 32
    int numRowsA = 16;
    int numColsA = 512;
    int numColsB = 32;

    size_t matrixSizeA = numRowsA * numColsA * sizeof(int8_t);
    size_t matrixSizeB = numColsA * numColsB * sizeof(int8_t);
    size_t resultSize = numRowsA * numColsB * sizeof(int8_t);

    // 在cpu上分配内存
    int8_t *h_matrixA, *h_matrixB, *h_result;
    h_matrixA = (int8_t *)malloc(matrixSizeA);
    h_matrixB = (int8_t *)malloc(matrixSizeB);
    h_result = (int8_t *)malloc(resultSize);

    // 随机初始化
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (size_t i = 0; i < matrixSizeA; ++i) {
        h_matrixA[i] = static_cast<int8_t>(std::rand() % 10);
    }
    for (size_t i = 0; i < matrixSizeB; ++i) {
        h_matrixB[i] = static_cast<int8_t>(std::rand() % 10);
    }

    // 在gpu上分配内存
    int8_t *d_matrixA, *d_matrixB, *d_result;
    checkCudaErrors(cudaMalloc(&d_matrixA, matrixSizeA));
    checkCudaErrors(cudaMalloc(&d_matrixB, matrixSizeB));
    checkCudaErrors(cudaMalloc(&d_result, resultSize));

    // h2d
    checkCudaErrors(cudaMemcpy(d_matrixA, h_matrixA, matrixSizeA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_matrixB, h_matrixB, matrixSizeB, cudaMemcpyHostToDevice));

    int block_size = 16
    dim3 threadsPerBlock(block_size, block_size);
    dim3 numBlocks(numColsB / threadsPerBlock.x, numRowsA / threadsPerBlock.y);

    // kernel launch!
    matrixDotProduct<<<numBlocks, threadsPerBlock>>>(d_matrixA, d_matrixB, d_result, numRowsA, numColsA, numColsB);

    // 检查内核执行错误
    checkCudaErrors(cudaGetLastError());

    // d2h
    checkCudaErrors(cudaMemcpy(h_result, d_result, resultSize, cudaMemcpyDeviceToHost));

    std::cout << "Matrix Dot Product Result:" << std::endl;
    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numColsB; ++j) {
            std::cout << static_cast<int>(h_result[i * numColsB + j]) << " ";
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
