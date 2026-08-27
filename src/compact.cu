#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void createFlags(
    const int *input,
    int *flags,
    int N)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N)
    {
        flags[i] = (input[i] % 2 == 0);
    }
}

__global__ void scanKernel(
    const int *input,
    int *output,
    int N)
{
    __shared__ int data[256];

    int tid = threadIdx.x;

    if (tid < N)
    {
        data[tid] = input[tid];
    }
    else
    {
        data[tid] = 0;
    }

    __syncthreads();

    for (int offset = 1;
         offset < blockDim.x;
         offset *= 2)
    {
        int value = 0;

        if (tid >= offset)
        {
            value = data[tid - offset];
        }

        __syncthreads();

        data[tid] += value;

        __syncthreads();
    }

    if (tid < N)
    {
        output[tid] = data[tid];
    }
}

__global__ void compact(
    const int *input,
    const int *flags,
    const int *scan,
    int *output,
    int N)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N && flags[i])
    {
        int outputIndex = scan[i] - 1;

        output[outputIndex] =
            input[i];
    }
}

int main()
{
    const int N = 8;

    int h_input[N] =
    {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    printf("Input:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_input[i]);
    }

    printf("\n");

    int h_output[N];

    int *d_input;
    int *d_flags;
    int *d_scan;
    int *d_output;

    size_t size = N * sizeof(int);

    CUDA_CHECK(cudaMalloc(
        &d_input, size));

    CUDA_CHECK(cudaMalloc(
        &d_flags, size));

    CUDA_CHECK(cudaMalloc(
        &d_scan, size));

    CUDA_CHECK(cudaMalloc(
        &d_output, size));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input,
        size,
        cudaMemcpyHostToDevice));

    int threads = 256;

    // ----------------------------
    // Step 1: Create flags
    // ----------------------------

    createFlags<<<1, threads>>>(
        d_input,
        d_flags,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // Step 2: Scan flags
    // ----------------------------

    scanKernel<<<1, threads>>>(
        d_flags,
        d_scan,
        N);

    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_scan,
        size,
        cudaMemcpyDeviceToHost));

    printf("Scan:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    }

    printf("\n");

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ----------------------------
    // Step 3: Compact
    // ----------------------------

    compact<<<1, threads>>>(
        d_input,
        d_flags,
        d_scan,
        d_output,
        N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_output,
        d_output,
        size,
        cudaMemcpyDeviceToHost));

    

    printf("Compacted:\n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    }

    printf("\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}