#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cuda_utils.cuh"

struct Point
{
    float x;
    float y;
    float z;
    float intensity;
};


void generatePointCloud(Point *points, int N)
{
    for (int i = 0; i < N; i++)
    {
        points[i].x = (float)(i %100) - 50.0f;
        points[i].y = (float)((i/100) % 100) - 50.0f;
        points[i].z = (float)(i%20)-10.0f;
        points[i].intensity = (float)(i % 255); 
    }    
}

__global__ void rangeFilter(const Point *points, int *flags, int N, float maxRange)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i <N)
    {
        Point p = points[i];

        float distanceSquared = p.x *p.x + p.y * p.y + p.z * p.z;
        flags[i] = (distanceSquared <= maxRange*maxRange);
    }
}

__global__ void compactPoints(
    const Point *points,
    const int *flags,
    const int *positions,
    Point *output,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i<N && flags[i])
    {
        int outputIndex = positions[i];
        output[outputIndex] = points[i];
    }
}

int main()
{
    const int N = 10000;
    const float maxRange = 3.0f;
    size_t pointSize= N * sizeof(Point);
    size_t flagSize = N * sizeof(int);

    Point *h_points = new Point[N];
    // int *h_flags = new int[N];

    generatePointCloud(h_points, N);

    printf("h_points: (%f,%f,%f,%f) \n",h_points[750].x,h_points[750].y,h_points[750].z,h_points[750].intensity);

    Point *d_points;
    int *d_flags;
    int *d_positions;
    Point *d_compacted;

    CUDA_CHECK(cudaMalloc(&d_points, pointSize));
    CUDA_CHECK(cudaMalloc(&d_flags, flagSize));
    CUDA_CHECK(cudaMalloc(&d_positions, flagSize));
    CUDA_CHECK(cudaMalloc(&d_compacted, pointSize));

    CUDA_CHECK(cudaMemcpy(d_points, h_points, pointSize, cudaMemcpyHostToDevice));
    
    int threadPerBlock = 256;
    int blocksPerGrid = (N + threadPerBlock + 1) /threadPerBlock;

    rangeFilter<<<blocksPerGrid,threadPerBlock>>>(d_points, d_flags, N, maxRange);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::device_ptr<int> flagsPtr(d_flags);
    thrust::device_ptr<int> positionsPtr(d_positions);

    thrust::exclusive_scan(flagsPtr, flagsPtr+N, positionsPtr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int lastPosition;
    int lastFlag;

    CUDA_CHECK(cudaMemcpy(&lastPosition, d_positions+N-1,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastFlag, d_flags+N-1, sizeof(int), cudaMemcpyDeviceToHost));
    
    int validCount = lastPosition + lastFlag;

    compactPoints<<<blocksPerGrid,threadPerBlock>>>(d_points, d_flags, d_positions, d_compacted, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // CUDA_CHECK(cudaMemcpy(h_flags, d_flags, flagSize, cudaMemcpyDeviceToHost));

    Point *h_compacted = new Point[validCount];
    if (validCount > 0)
    {
        CUDA_CHECK(cudaMemcpy( h_compacted, d_compacted, validCount * sizeof(Point), cudaMemcpyDeviceToHost));
    }

    printf("Total points: %d \n", N);
    printf("Valid points: %d \n", validCount);
    printf("Removed points: %d \n", N-validCount);

    int printCount =
        validCount < 5
        ? validCount
        : 5;

    for (int i = 0;
         i < printCount;
         i++)
    {
        printf(
            "P%d = (%f, %f, %f, %f)\n",
            i,
            h_compacted[i].x,
            h_compacted[i].y,
            h_compacted[i].z,
            h_compacted[i].intensity);
    }


    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_compacted));

    delete[] h_points;
    delete[] h_compacted;
    // delete[] h_flags;

    return 0;
}