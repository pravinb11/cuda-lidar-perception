#include <stdio.h>
#include <float.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

float reduceSumCPU(
    const float *data,
    int N)
{
    float sum = 0.0f;

    for (int i = 0; i < N; i++)
    {
        sum += data[i];
    }

    return sum;
}



// int main()
// {
//     const int N = 1 << 20;

//     float *h_data = new float[N];

//     for (int i = 0; i < N; i++)
//     {
//         h_data[i] = 1.0f;
//     }

//     float result = reduceSumCPU(
//         h_data,
//         N);

//     printf("CPU sum = %.1f\n", result);

//     delete[] h_data;

//     return 0;
// }

__device__ float warpReduceSum(float value)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void reduceSumOptimized(const float *input, float *output, int N)
{
    __shared__ float warpSums[32];

    int tid = threadIdx.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    if (i<N)
    {
        value = input[i];
    }

    // Step 1: Reduce within each warp

    value = warpReduceSum(value);

    // Step 2: One thread per warp stores its result

    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    if(lane == 0)
    {
        warpSums[warpId] = value;
    }

    __syncthreads();

    // Step 3: First warp reduces the warp results

    value = 0.0f;
    if(warpId == 0)
    {
        int numWarps = (blockDim.x + warpSize -1) / warpSize;

        if (lane < numWarps)
        {
            value = warpSums[lane];
        }

        value = warpReduceSum(value);

        if (lane == 0)
        {
            output[blockIdx.x] = value;
        }
    }
}

__global__ void reduceSumKernel(
    const float *input,
    float *output,
    int N)
{
    __shared__ float sharedData[256];

    int tid = threadIdx.x;

    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

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
        output[blockIdx.x] =
            sharedData[0];
    }
}

int main()
{
    const int N = 1 << 20;

    const size_t size =
        N * sizeof(float);

    float *h_data =
        new float[N];

    for (int i = 0; i < N; i++)
    {
        h_data[i] = 1.0f;
    }

    // -----------------------------
    // Allocate device input
    // -----------------------------

    float *d_input;

    CUDA_CHECK(cudaMalloc(
        &d_input,
        size));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_data,
        size,
        cudaMemcpyHostToDevice));

    // -----------------------------
    // Number of blocks
    // -----------------------------

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    // -----------------------------
    // Partial results
    // -----------------------------

    float *d_partial;

    CUDA_CHECK(cudaMalloc(
        &d_partial,
        blocksPerGrid * sizeof(float)));

    // -----------------------------
    // Reduction
    // -----------------------------

    // reduceSumKernel<<<
    //     blocksPerGrid,
    //     threadsPerBlock
    // >>>(
    //     d_input,
    //     d_partial,
    //     N
    // );

    reduceSumOptimized<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_input,
        d_partial,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // -----------------------------
    // Copy partial sums
    // -----------------------------

    float *h_partial =
        new float[blocksPerGrid];

    CUDA_CHECK(cudaMemcpy(
        h_partial,
        d_partial,
        blocksPerGrid * sizeof(float),
        cudaMemcpyDeviceToHost));

    // -----------------------------
    // Final reduction on CPU
    // -----------------------------

    float gpu_result = 0.0f;

    for (int i = 0;
         i < blocksPerGrid;
         i++)
    {
        gpu_result += h_partial[i];
    }

    printf("GPU sum = %.1f\n",
           gpu_result);

    // -----------------------------
    // Cleanup
    // -----------------------------

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_partial));

    delete[] h_data;
    delete[] h_partial;

    return 0;
}