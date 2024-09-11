#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

void checkCudaErrors(cudaError_t err) {
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 对图片矩阵进行归一化（减均值除以方差）
__global__ void NormalizeImage(float *d_imageData, float d_mean, float d_scale, int d_numPixels, float *d_normalizeExecutionTime) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < d_numPixels) {
        d_imageData[index] = (d_imageData[index] - d_mean) / d_scale;
    }
    cudaEventRecord(*d_normalizeExecutionTime);
}

// 对图片矩阵进行转置
__global__ void TransposeImage(float *d_input, float *d_output, int d_width, int d_height, float *d_transposeExecutionTime) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < d_width && y < d_height) {
        d_output[y * d_width + x] = d_input[x * d_height + y];
    }
    cudaEventRecord(*d_transposeExecutionTime);
}

// 对两个矩阵进行点乘
__global__ void DotProduct(float *d_matrix1, float *d_matrix2, float *d_result, int d_numPixels) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < d_numPixels) {
        d_result[index] = d_matrix1[index] * d_matrix2[index];
    }
}

int main() {
    int d_width = 1024;
    int d_height = 1024;
    int d_numPixels = d_width * d_height;

    float *h_imageData, *h_transposedImage, *h_dotProductResult;
    float *d_imageData, *d_transposedImage, *d_dotProductResult;

    // 分配cpu内存
    h_imageData = (float *)malloc(d_numPixels * sizeof(float));
    h_transposedImage = (float *)malloc(d_numPixels * sizeof(float));
    h_dotProductResult = (float *)malloc(d_numPixels * sizeof(float));

    // 随机初始化图片矩阵
    for (int i = 0; i < d_numPixels; ++i) {
        h_imageData[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 在gpu上分配内存
    checkCudaErrors(cudaMalloc(&d_imageData, d_numPixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_transposedImage, d_numPixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_dotProductResult, d_numPixels * sizeof(float)));

    // h2d
    checkCudaErrors(cudaMemcpy(d_imageData, h_imageData, d_numPixels * sizeof(float), cudaMemcpyHostToDevice));

    float d_mean = 128.0f;  
    float d_scale = 0.4918f;  

    dim3 threadsPerBlock(256);
    dim3 numBlocks(d_numPixels / threadsPerBlock.x);

    // 记录kernel执行时间的事件
    cudaEvent_t d_normalizeTime, d_transposeTime, d_dotProductTime;
    checkCudaErrors(cudaEventCreate(&d_normalizeTime));
    checkCudaErrors(cudaEventCreate(&d_transposeTime));
    checkCudaErrors(cudaEventCreate(&d_dotProductTime));

    // 异步launch 归一化
    cudaStream_t d_normalizeStream;
    checkCudaErrors(cudaStreamCreate(&d_normalizeStream));
    float *d_normalizeExecutionTime;
    checkCudaErrors(cudaMalloc(&d_normalizeExecutionTime, sizeof(float)));
    checkCudaErrors(NormalizeImage<<<numBlocks, threadsPerBlock, 0, d_normalizeStream>>>(d_imageData, d_mean, d_scale, d_numPixels, d_normalizeExecutionTime));

    // 异步launch 转置
    cudaStream_t d_transposeStream;
    checkCudaErrors(cudaStreamCreate(&d_transposeStream));
    float *d_transposeExecutionTime;
    checkCudaErrors(cudaMalloc(&d_transposeExecutionTime, sizeof(float)));
    checkCudaErrors(TransposeImage<<<numBlocks, threadsPerBlock, 0, d_transposeStream>>>(d_imageData, d_transposedImage, d_width, d_height, d_transposeExecutionTime));

    // 等待两个异步操作结束
    checkCudaErrors(cudaStreamSynchronize(d_normalizeStream));
    checkCudaErrors(cudaStreamSynchronize(d_transposeStream));

    dim3 threadsPerBlockDotProduct(256);
    dim3 numBlocksDotProduct(d_numPixels / threadsPerBlockDotProduct.x);

    // 拿取结果进行点乘
    cudaStream_t d_dotProductStream;
    checkCudaErrors(cudaStreamCreate(&d_dotProductStream));
    checkCudaErrors(DotProduct<<<numBlocksDotProduct, threadsPerBlockDotProduct, 0, d_dotProductStream>>>(d_imageData, d_transposedImage, d_dotProductResult, d_numPixels));

    checkCudaErrors(cudaEventRecord(d_dotProductTime));
    checkCudaErrors(cudaStreamSynchronize(d_dotProductStream));

    // d2h
    checkCudaErrors(cudaMemcpy(h_imageData, d_imageData, d_numPixels * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_transposedImage, d_transposedImage, d_numPixels * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_dotProductResult, d_dotProductResult, d_numPixels * sizeof(float), cudaMemcpyDeviceToHost));

    // 计算耗时
    float normalizeElapsedTime, transposeElapsedTime, dotProductElapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&normalizeElapsedTime, d_normalizeTime, d_normalizeExecutionTime));
    checkCudaErrors(cudaEventElapsedTime(&transposeElapsedTime, d_transposeTime, d_transposeExecutionTime));
    checkCudaErrors(cudaEventElapsedTime(&dotProductElapsedTime, d_dotProductTime, NULL));

    std::cout << "Normalize time: " << normalizeElapsedTime << " ms" << std::endl;
    std::cout << "Transpose time: " << transposeElapsedTime << " ms" << std::endl;
    std::cout << "Dot Product time: " << dotProductElapsedTime << " ms" << std::endl;

    // 释放资源
    checkCudaErrors(cudaFree(d_imageData));
    checkCudaErrors(cudaFree(d_transposedImage));
    checkCudaErrors(cudaFree(d_dotProductResult));
    checkCudaErrors(cudaFree(d_normalizeExecutionTime));
    checkCudaErrors(cudaFree(d_transposeExecutionTime));
    free(h_imageData);
    free(h_transposedImage);
    free(h_dotProductResult);
    checkCudaErrors(cudaEventDestroy(d_normalizeTime));
    checkCudaErrors(cudaEventDestroy(d_transposeTime));
    checkCudaErrors(cudaEventDestroy(d_dotProductTime));
    checkCudaErrors(cudaStreamDestroy(d_normalizeStream));
    checkCudaErrors(cudaStreamDestroy(d_transposeStream));
    checkCudaErrors(cudaStreamDestroy(d_dotProductStream));

    // 检查 CUDA error
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaDeviceReset();

    return 0;
}