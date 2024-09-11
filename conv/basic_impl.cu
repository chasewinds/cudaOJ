#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 

void checkCudaErrors(cudaError_t err) {
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 简易的卷积计算conv的实现（不考虑 padding）
// 尽可能的实现 使用tensor core进行计算
__global__ void convHalfWithoutPadding(
    half *input, 
    half *kernel, 
    half *output, 
    int kernelSize, 
    int stride, 
    int inChannels, 
    int imageHeight, 
    int imageWidth
) {
    int outChannel = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < imageHeight && col < imageWidth) { // 边界检查
        half sum = 0;
        for (int inChannel = 0; inChannel < inChannels; ++inChannel) {
            for (int kRow = 0; kRow < kernelSize; ++kRow) {
                for (int kCol = 0; kCol < kernelSize; ++kCol) {
                    int inputRow = row + kRow * stride;
                    int inputCol = col + kCol * stride;

                    if (inputRow >= 0 && inputRow < imageHeight && inputCol >= 0 && inputCol < imageWidth) { // 滑动窗口边界检查
                        sum += input[(inChannel * imageHeight + inputRow) * imageWidth + inputCol] * 
                               kernel[(outChannel * inChannels + inChannel) * kernelSize * kernelSize + kRow * kernelSize + kCol];
                    }
                }
            }
        }
        output[(outChannel * imageHeight + row) * imageWidth + col] = sum;
    }
}

// 测试
void testConvolution(
    int kernelSize, 
    int stride, 
    int inChannels
) {
    int imageHeight = 128;
    int imageWidth = 128;

    half *h_input, 
        *h_kernel, 
        *h_output;
    size_t inputSize = imageHeight * imageWidth * inChannels * sizeof(half);
    size_t kernelSizeBytes = inChannels * kernelSize * kernelSize * sizeof(half);
    size_t outputSize = imageHeight * imageWidth * sizeof(half);

    // 预分配cpu内存
    h_input = (half *)malloc(inputSize);
    h_kernel = (half *)malloc(kernelSizeBytes);
    h_output = (half *)malloc(outputSize);

    // 随机初始化 img和卷积核
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (size_t i = 0; i < inputSize; ++i) {
        h_input[i] = static_cast<half>(std::rand() / static_cast<float>(RAND_MAX));
    }
    for (size_t i = 0; i < kernelSizeBytes; ++i) {
        h_kernel[i] = static_cast<half>(std::rand() / static_cast<float>(RAND_MAX));
    }

    // 分配device内存
    half *d_input, 
        *d_kernel, 
        *d_output;
    cudaError_t err;
    err = cudaMalloc(&d_input, inputSize);
    checkCudaErrors(err);
    err = cudaMalloc(&d_kernel, kernelSizeBytes);
    checkCudaErrors(err);
    err = cudaMalloc(&d_output, outputSize);
    checkCudaErrors(err);

    // h2d 拷贝到gpu
    err = cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    checkCudaErrors(err);
    err = cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice);
    checkCudaErrors(err);

    // 假设使用A10， 40个SM单元
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                    (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    // kernel launch！
    convHalfWithoutPadding<<<numBlocks, threadsPerBlock>>>(
        d_input, d_kernel, d_output, kernelSize, stride, inChannels, imageHeight, imageWidth
    );

    // 检查错误
    checkCudaErrors(cudaGetLastError());

    // d2h 拷贝回cpu
    err = cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    checkCudaErrors(err);

    // 打印部分结果
    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<float>(h_output[i]) << " ";
    }
    std::cout << std::endl;

    // 释放gpu内存
    err = cudaFree(d_input);
    checkCudaErrors(err);
    err = cudaFree(d_kernel);
    checkCudaErrors(err);
    err = cudaFree(d_output);
    checkCudaErrors(err);
    // 释放cpu内存
    free(h_input);
    free(h_kernel);
    free(h_output);
}

int main() {
    int kernelSize = 3;
    int stride = 1;
    int inChannels = 3;

    testConvolution(kernelSize, stride, inChannels);

    return 0;
}