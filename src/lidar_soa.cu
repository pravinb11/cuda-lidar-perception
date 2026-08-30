#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cuda_utils.cuh"

struct PointCloudSoA
{
    float *x;
    float *y;
    float *z;
    float *intensity;
};

__global__ void rangeFilterSoA(
    const float *x,
    const float *y,
    const float *z,
    int *flags,
    int N,
    float maxRange)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < N)
    {
        float distanceSquared =
            x[i] * x[i] +
            y[i] * y[i] +
            z[i] * z[i];

        flags[i] =
            distanceSquared <=
            maxRange * maxRange;
    }
}

int main()
{
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_z = new float[N];
    float *h_intensity = new float[N];


    for (int i = 0; i < N; i++)
    {
        h_x[i] = (float) (i%100) - 50.0f;
        h_y[i] = (float) ((i/100) % 100) - 50.0f;
        h_z[i] = (float) (i%20) - 10.0f;
        h_intensity[i] = (float)(i%255); 
    }

    float *d_x;
    float *d_y;
    float *d_z;
    float *d_intensity;

    CUDA_CHECK(cudaMalloc(
        &d_x,
        N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_y,
        N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_z,
        N * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &d_intensity,
        N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_x,
        h_x,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_y,
        h_y,
        N * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_z,
        h_z,
        N * sizeof(float),
        cudaMemcpyHostToDevice));
}

