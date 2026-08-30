#include <stdio.h>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <vector>

#include "cuda_utils.cuh"
#include "point.h"
#include "pointcloud_loader.h"
#include "cuda_timer.cuh"


// CUDA Kernel 1: Range Filter
__global__ void rangeFilter(const Point *points, int *flags, int N, float maxRange)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<N)
    {
        Point p = points[i];

        float distanceSquared = p.x * p.x + p.y * p.y +p.z * p.z;

        flags[i] = (distanceSquared <= maxRange*maxRange);
    }
}

// CUDA Kernel 2: Compact Valid Points
__global__ void compactPoints(const Point *points, const int *flags, const int *positions, Point *output, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N && flags[i])
    {
        int outputIndex = positions[i];
        output[outputIndex] = points[i];
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Error, need to give arguments: %s <file.pcd> \n", argv[0]);

        return 1;
    }
    else
    {
        printf("Received file: %s \n", argv[1] );
    }
    
    const char *filename = argv[1];

    const float maxRange = 15.0f;

    // Load PCD
    std::vector<Point> points;
    if(!loadPCD(filename, points))
    {
        return 1;
    }

    int N = static_cast<int>(points.size());

    if (N==0)
    {
        printf("Point cloud is empty \n");
        return 1;
    }

    printf("Input points: %d \n",N);

    size_t pointSize = N * sizeof(Point);
    size_t intSize = N * sizeof(int);

    Point *d_points = nullptr;
    int *d_flags = nullptr;
    int *d_positions = nullptr;
    Point *d_compacted = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_points, pointSize));
    CUDA_CHECK(cudaMalloc(&d_flags, intSize));
    CUDA_CHECK(cudaMalloc(&d_positions, intSize));
    CUDA_CHECK(cudaMalloc(&d_compacted, pointSize));

    CudaTimer h2dTimer;
    h2dTimer.begin();
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), pointSize, cudaMemcpyHostToDevice));
    float h2dTime = h2dTimer.end();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N+ threadsPerBlock -1) / threadsPerBlock;
    printf(
        "Threads per block: %d\n",
        threadsPerBlock);

    printf(
        "Blocks: %d\n",
        blocksPerGrid);
    CudaTimer filterTimer;

    filterTimer.begin();
    rangeFilter<<<blocksPerGrid,threadsPerBlock>>>(d_points, d_flags, N, maxRange);
    float filterTime = filterTimer.end();
    
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaDeviceSynchronize());
    
    // STEP 2: Exclusive Scan
    
    thrust::device_ptr<int> flagsPtr(d_flags);
    thrust::device_ptr<int> positionsPtr(d_positions);
    CudaTimer scanTimer;

    scanTimer.begin();

    thrust::exclusive_scan(flagsPtr, flagsPtr+N, positionsPtr);

    CUDA_CHECK(cudaDeviceSynchronize());
    float scanTime = scanTimer.end();

    // Step 3: Calculate Valid Point Count

    int lastPosition = 0;
    int lastFlag = 0;

    CUDA_CHECK(cudaMemcpy(&lastPosition, d_positions + N -1, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&lastFlag, d_flags + N -1, sizeof(int),cudaMemcpyDeviceToHost));
    int validCount = lastPosition + lastFlag;

    // Step4: GPU Point Compaction
    CudaTimer compactTimer;
    compactTimer.begin();
    compactPoints<<<blocksPerGrid, threadsPerBlock>>>(d_points, d_flags, d_positions, d_compacted, N);
    float compactTime = compactTimer.end();
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaDeviceSynchronize());

    // STEP 5: Allocate Host Output
    // ========================================================

    std::vector<Point>
        filteredPoints(validCount);

    // ========================================================
    // STEP 6: Copy Compact Point Cloud
    //
    // GPU → CPU
    // ========================================================
    CudaTimer d2hTimer;

    d2hTimer.begin();
    if (validCount > 0)
    {
        CUDA_CHECK(cudaMemcpy(
            filteredPoints.data(),
            d_compacted,
            validCount * sizeof(Point),
            cudaMemcpyDeviceToHost));
    }
    float d2hTime = d2hTimer.end();
    // Results
    // ========================================================

    printf("\n");
    printf("============================\n");
    printf("       CUDA RESULTS\n");
    printf("============================\n");

    printf(
        "Total points   : %d\n",
        N);

    printf(
        "Valid points   : %d\n",
        validCount);

    printf(
        "Removed points : %d\n",
        N - validCount);

    printf(
        "Max range      : %.2f\n",
        maxRange);

    // --------------------------------------------------------
    // Print first few filtered points
    // --------------------------------------------------------

    printf("\nFirst filtered points:\n");

    int printCount =
        validCount < 5
        ? validCount
        : 5;

    for (int i = 0;
         i < printCount;
         i++)
    {
        printf(
            "P%d = (%f, %f, %f)\n",
            i,
            filteredPoints[i].x,
            filteredPoints[i].y,
            filteredPoints[i].z);
    }

    CUDA_CHECK(cudaFree(
        d_points));

    CUDA_CHECK(cudaFree(
        d_flags));

    CUDA_CHECK(cudaFree(
        d_positions));

    CUDA_CHECK(cudaFree(
        d_compacted));

    printf("\n");
printf("============================\n");
printf("       PERFORMANCE\n");
printf("============================\n");

printf(
    "H2D transfer : %.3f ms\n",
    h2dTime);

printf(
    "Range filter : %.3f ms\n",
    filterTime);

printf(
    "Scan         : %.3f ms\n",
    scanTime);

printf(
    "Compaction   : %.3f ms\n",
    compactTime);

printf(
    "D2H transfer : %.3f ms\n",
    d2hTime);

float gpuProcessingTime =
    filterTime +
    scanTime +
    compactTime;

printf(
    "GPU kernels  : %.3f ms\n",
    gpuProcessingTime);
    return 0;
}