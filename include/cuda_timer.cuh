#pragma once 
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

class CudaTimer
{
public:

    cudaEvent_t start;
    cudaEvent_t stop;

    CudaTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void begin()
    {
        CUDA_CHECK(cudaEventRecord(start));
    }

    float end()
    {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        return milliseconds;
    }
};