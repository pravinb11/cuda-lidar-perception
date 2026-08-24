#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloFromGPU()
{
    printf("Hello from GPU! Thread %d \n", threadIdx.x);
}

int main()
{
    printf("Hello from CPU! \n");
    helloFromGPU<<<1,5>>>();

    cudaError_t error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("Kernel launch error: %s \n", cudaGetErrorString(error));
        return 1;
    }
    cudaDeviceSynchronize();
    error = cudaGetLastError();

    if(error != cudaSuccess)
    {
        printf("Kernel execution error: %s \n", cudaGetErrorString(error));
        return 1;
    }

    printf("CUDA execution successful! \n");

    return 0;
}