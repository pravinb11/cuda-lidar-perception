#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void vectorAddKernel(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 24;
    const size_t size = N * sizeof(float);

    printf("Number of elements: %d\n", N);
    printf("Data size: %.2f MB\n",
           size / (1024.0 * 1024.0));

    // ----------------------------
    // Host memory
    // ----------------------------

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ----------------------------
    // Device memory
    // ----------------------------

    float *d_A;
    float *d_B;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // ----------------------------
    // Host → Device
    // ----------------------------

    cudaEvent_t h2d_start, h2d_stop;

    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));

    CUDA_CHECK(cudaEventRecord(h2d_start));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        size,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        size,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(h2d_stop));

    float h2d_ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &h2d_ms,
        h2d_start,
        h2d_stop));

    // ----------------------------
    // Kernel
    // ----------------------------

    int threadsPerBlock = 256;

    int blocksPerGrid =
        (N + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaEvent_t kernel_start, kernel_stop;

    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    CUDA_CHECK(cudaEventRecord(kernel_start));

    vectorAddKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_C,
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));

    float kernel_ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &kernel_ms,
        kernel_start,
        kernel_stop));

    // ----------------------------
    // Device → Host
    // ----------------------------

    cudaEvent_t d2h_start, d2h_stop;

    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));

    CUDA_CHECK(cudaEventRecord(d2h_start));

    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));

    float d2h_ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &d2h_ms,
        d2h_start,
        d2h_stop));

    // ----------------------------
    // Results
    // ----------------------------

    printf("\nPerformance:\n");

    printf("H -> D: %.3f ms\n", h2d_ms);
    printf("Kernel: %.3f ms\n", kernel_ms);
    printf("D -> H: %.3f ms\n", d2h_ms);

    printf(
        "Total: %.3f ms\n",
        h2d_ms + kernel_ms + d2h_ms
    );

    // ----------------------------
    // Validation
    // ----------------------------

    printf("\nValidation:\n");

    printf("C[0] = %.1f\n", h_C[0]);
    printf("C[N-1] = %.1f\n", h_C[N - 1]);

    // ----------------------------
    // Cleanup
    // ----------------------------

    CUDA_CHECK(cudaEventDestroy(h2d_start));
    CUDA_CHECK(cudaEventDestroy(h2d_stop));

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    CUDA_CHECK(cudaEventDestroy(d2h_start));
    CUDA_CHECK(cudaEventDestroy(d2h_stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}