#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#define N 9

__global__ void max_sub_array(int *nums, int *d_res, int n) {
    int max_value = nums[0];
    int ma = 0;
    for(int i = 0; i < n; i++) {
	ma = max(ma + nums[i], nums[i]);
	max_value = max(max_value, ma);
    }	
    d_res[0] = max_value;
}

int main() {
    int *nums;
    int *res;

    int *d_nums;
    int *d_res;

    // allocate host mem 
    nums = (int*)malloc(sizeof(int) * N);
    int input_nums[N] = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    for(int i = 0; i < N; i++) {
	nums[i] = input_nums[i];
    }

    res = (int*)malloc(sizeof(int));

    // allocate device mem
    cudaMalloc((void**)&d_nums, sizeof(int) * N);
    cudaMalloc((void**)&d_res, sizeof(int) * N);


    // h2d
    cudaMemcpy(d_nums, nums, sizeof(int) * N, cudaMemcpyHostToDevice);

    // launch kernel 
    max_sub_array<<<1, 1>>>(d_nums, d_res, N);

    // d2h
    cudaMemcpy(&res, d_res, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", res);

    // deallocate device memory
    cudaFree(d_res);
    cudaFree(d_nums);
    printf("%s\n", "cuda free");

    // deallocate host memory
    free(nums);
    //free(res);
    printf("%s\n", "cpu free");

    return 0;
}
