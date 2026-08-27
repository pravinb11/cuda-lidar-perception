#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.cuh"

__global__ void helloKernel()
{
    printf("Hello from GPU!\n");
}

__global__ void errorKernel(int *data)
{
    int i = threadIdx.x;
    data[1] = 42;
}

int main()
{
    printf("Starting CUDA program...\n");

    int *d_data;

    // Allocate memory for 5 integers
    CUDA_CHECK(cudaMalloc(&d_data, 5 * sizeof(int)));
    // cudaMalloc(&d_data, 5 * sizeof(int));
    // Allocate 5 integers but launch 10 threads
    errorKernel<<<1, 10>>>(d_data);

    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    // cudaDeviceSynchronize();

    printf("CUDA kernel completed successfully.\n");

    CUDA_CHECK(cudaFree(d_data));
    // cudaFree(d_data);

    return 0;
}