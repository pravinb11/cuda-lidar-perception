#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

void prefixSumCPU(const int *input, int *output, int N)
{
    output[0] = input[0];

    for (int i = 1; i < N; i++)
    {
        output[i] = output[i-1] + input[i];
    }
}

__global__ void scanKernel(const int *input, int *output, int N)
{
    __shared__ int data[256];

    int tid = threadIdx.x;

    if(tid < N)
    {
        data[tid] = input[tid];
    }
    else
    {
        data[tid] = 0;
    }

    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int value = 0;

        if(tid >= offset)
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

int main()
{
    const int N = 8;

    int input[N] = {1,2,3,4,5,6,7,8};

    int output[N];

    prefixSumCPU(input, output, N);

    printf("Prefix sum: \n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", output[i]);
    }

    printf("\n");

    int h_output[N];

    int *d_input;
    int *d_output;

    CUDA_CHECK(cudaMalloc(&d_input, N* sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N* sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, input, N *sizeof(float), cudaMemcpyHostToDevice));
    scanKernel<<<1,256>>>(d_input, d_output,N);
    CUDA_CHECK(cudaMemcpy(h_output,d_output, N*sizeof(float), cudaMemcpyDeviceToHost ));
    
    printf("GPU prefix sum : \n");

    for (int i = 0; i < N; i++)
    {
        printf("%d ", h_output[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}