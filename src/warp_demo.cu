#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void divergenceDemo()
{
    int tid = threadIdx.x;

    if (tid % 2 == 0)
    {
        printf("Thread %d -> EVEN\n", tid);
    }
    else
    {
        printf("Thread %d -> ODD\n", tid);
    }
}

__global__ void warpInfo()
{
    printf(
        "Thread %d, warp size = %d\n",
        threadIdx.x,
        warpSize
    );
}
__device__ float warpReduceSum(float value)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_sync(0xffffffff, value, offset);
    }
    return value;
}
__global__ void shuffleDemo()
{
    int lane = threadIdx.x % warpSize;

    int warpId = threadIdx.x / warpSize;
    

    int value = lane * 1;

    int valueFromLane0 = __shfl_sync(0xffffffff, value, 1);
    

    float sum = warpReduceSum(value);
    printf("Lane %d: value: %d, lane0: %d, Sum: %f \n", lane, value, valueFromLane0, sum);
    
}



int main()
{
    // divergenceDemo<<<1, 32>>>();
    // warpInfo<<<1, 64>>>();
    shuffleDemo<<<1,32>>>();
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}