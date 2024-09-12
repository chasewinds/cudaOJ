#include <stdio.h>
#include <assert.n>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// cuda helper func and utils 
#include <helper_function.h>
#include <helper_cuda.h>

// 学习cuda教程官方demo的Mul实现 https://github.com/NVIDIA/cuda-samples/blob/master/Samples/0_Introduction/matrixMul/matrixMul.cu
// dimA: 320 * 320; wA = 320, hA = 320;
// dimB: 640 * 320; wB = 640, hB = 320;
// BLOCK_SIZE: 16 | 32; 
template <int BLOCK_SIZE> 
// 为什么函数签名里只有m， n没有k？分块计算，BLOCK_SIZE = k
__global__ void Mul2DMatrix(float* C, float* A, float* B, int wA, int wB) { 
    // aBegin ~ aEnd loop by row order
    int aBegin = wA * (BLOCK_SIZE * blockIdx.y); // threads.y * block_size max= dimsA.y = hA = 320?
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;

    // bBegin: every col begin idx
    int bBegin = BLOCK_SIZE * blockIdx.x; // threads.x * block_size max= dimsB.x = wB = 640?
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0; // 一行 * 一列的总加和，与BLOCK_SIZE 无关

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // a sub matrix shard memory arr
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // 把BLOCK_SIZE*BLOCK_SIZE range 里的对应内存段数据， 按照内存偏移，拷贝到共享矩阵As, Bs 对应row, col中
        As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];

        __syncthreads(); // 确保所有线程数据都加载完

        #pragma unroll // 编译时展开循环
        for (int k = 0; k < BLOCK_SIZE; k++) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // 行 * 列 + 和
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x; // block level offset
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

int main() {
    int block_size = 32;
    dim3 dimsA = (320, 320, 1);
    dim3 dimsB = (640, 320, 1);

    unsigned int size_A  = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A;
    checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));

    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B;
    checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));

    cudaStream_t stream;

    for (int i = 0; i < size_A; ++i) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < size_B; ++i) {
        h_B[i] = 0.01f;
    }

    float* d_A, *d_B, *d_C;

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));

    // if (h_C == NULL) {

    // }

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    checkCudaErrors(cudamemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudamemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));

    dim3 threads(block_size, block_size); // thread.idx
    // dimsB.x = 640, dimsA.y = 320
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y); // SM level block.idx

    // dimsA.x = 320, dimsB.x = 640
    // 预热
    Mul2DMatrix<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    // Execute the kernel
    int nIter = 300;

    for (int j = 0; j < nIter; j++) {
        Mul2DMatrix<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    checkCudaErrors(cudaEventRecord(stop, stream));

    checkCudaErrors(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    // Clean up memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    reutrn 0;
}
