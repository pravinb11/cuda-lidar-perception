#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void goodAccess(
    const float *input,
    float *output,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        output[i] = input[i] * 2.0f;
    }
}

int main()
{
    const int N = 1 << 24;
    const size_t size = N * sizeof(float);

    float *h_input = new float[N];
    float *h_output = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }

    float *d_input;
    float *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input,
        size,
        cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    goodAccess<<<blocksPerGrid, threadsPerBlock>>>(
        d_input,
        d_output,
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop));

    printf("Kernel time: %.3f ms\n",
           milliseconds);

    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        size,
        cudaMemcpyDeviceToHost));

    printf("Output[0] = %.1f\n",
           h_output[0]);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    delete[] h_input;
    delete[] h_output;

    return 0;
}