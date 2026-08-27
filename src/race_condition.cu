#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

// __global__ void incrementCounter(int *counter)
// {
//     *counter = *counter + 1;
// }

__global__ void incrementCounter(int *counter)
{
    atomicAdd(counter, 1);
}
int main()
{
    int h_counter = 0;
    int *d_counter;

    CUDA_CHECK(cudaMalloc(
        &d_counter,
        sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_counter,
        &h_counter,
        sizeof(int),
        cudaMemcpyHostToDevice));

    int threads = 1000;

    incrementCounter<<<1, threads>>>(
        d_counter);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        &h_counter,
        d_counter,
        sizeof(int),
        cudaMemcpyDeviceToHost));

    printf("Counter = %d\n", h_counter);

    CUDA_CHECK(cudaFree(d_counter));

    return 0;
}