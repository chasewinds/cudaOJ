#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h> 

void checkCudaErrors(cudaError_t err) 
{
    if (err!= cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void convHalfKernel(half *input, half *output, int kernelSize, int stride, int inChannels, int imageHeight, int imageWidth)
{

    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;


    if (globalRow < imageHeight && globalCol < imageWidth) {
        half sum = 0;
        for (int inChannel = 0; inChannel < inChannels; ++inChannel) {
            for (int kRow = 0; kRow < kernelSize; ++kRow) {
                for (int kCol = 0; kCol < kernelSize; ++kCol) {
                    int inputRow = globalRow + kRow * stride;
                    int inputCol = globalCol + kCol * stride;

                    if (inputRow >= 0 && inputRow < imageHeight && inputCol >= 0 && inputCol < imageWidth) {
                        sum += input[(inChannel * imageHeight + inputRow) * imageWidth + inputCol] * 
                               kernel[(inChannel * kernelSize * kernelSize) + (kRow * kernelSize + kCol)];
                    }
                }
            }
        }
        output[(globalRow * imageWidth + globalCol)] = sum;
    }
}

int main()
{
    int kernelSize = 3;
    int stride = 1;
    int inChannels = 3;
    int imageHeight = 128;
    int imageWidth = 128;


    half *h_input, *h_output;
    size_t inputSize = imageHeight * imageWidth * inChannels * sizeof(half);
    size_t outputSize = imageHeight * imageWidth * sizeof(half);

    h_input = (half *)malloc(inputSize);
    h_output = (half *)malloc(outputSize);


    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (size_t i = 0; i < inputSize; ++i) {
        h_input[i] = static_cast<half>(std::rand() / static_cast<float>(RAND_MAX));
    }


    half *d_input, *d_output;
    cudaError_t err = cudaMalloc(&d_input, inputSize);
    checkCudaErrors(err);
    err = cudaMalloc(&d_output, outputSize);
    checkCudaErrors(err);


    err = cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    checkCudaErrors(err);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // with thread block cluster, kernel launch!
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 4; 
        attribute[0].val.clusterDim.y = 4;
        attribute[0].val.clusterDim.z = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, convHalfKernel, d_input, d_output, kernelSize, stride, inChannels, imageHeight, imageWidth);
    }

    err = cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    checkCudaErrors(err);

    for (int i = 0; i < 10; ++i) {
        std::cout << static_cast<float>(h_output[i]) << " ";
    }
    std::cout << std::endl;

    err = cudaFree(d_input);
    checkCudaErrors(err);
    err = cudaFree(d_output);
    checkCudaErrors(err);
    free(h_input);
    free(h_output);

    return 0;
}
