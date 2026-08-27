#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void reduceSum(
    const float *input,
    float *output,
    int N)
{
    __shared__ float sharedData[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        sharedData[tid] = input[i];
    }
    else
    {
        sharedData[tid] = 0.0f;
    }

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            sharedData[tid] +=
                sharedData[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

int main()
{
    const int N = 8;

    float h_input[N] =
    {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    float h_output[1];

    float *d_input;
    float *d_output;

    CUDA_CHECK(cudaMalloc(
        &d_input,
        N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_output,
        sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    reduceSum<<<1, 256>>>(
        d_input,
        d_output,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost));

    printf("Sum = %.1f\n",
           h_output[0]);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}