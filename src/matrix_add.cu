#include <stdio.h>
#include<cuda_runtime.h>

#include "cuda_utils.cuh"

__global__ void matrixAdd(
    const float *A,
    const float *B,
    float *C,
    int rows,
    int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols)
    {
        int index = row * cols + col;

        C[index] = A[index] + B[index];

        printf("blockIdx.x: %d,  blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d, index: %d, A[%d, %d]: %2.1f, B[%d,%d]: %2.1f, C[%d,%d]: %2.1f \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, index, row,col, A[index], row,col, B[index], row,col, C[index]);
    }
}

int main()
{
    const int rows = 4;
    const int cols = 4;

    const int N = rows * cols;
    const size_t size = N * sizeof(float);

    // float h_A[N];
    // float h_B[N];
    // float h_C[N];
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; i++)
    {

        h_A[i] = i;
        h_B[i] = i+10;
        // printf("i: %d, h_A[i]: %f, h_B[i]: %f \n", i, h_A[i], h_B[i]);
    }

    printf("h_A \n");
    // for (int row = 0; row < rows; row++)
    // {
    //     for (int col = 0; col < cols; col++)
    //     {
    //         printf("%6.1f ", h_A[row * cols +col]);
    //     }
    //     printf("\n");
        
    // }

    printf("h_B \n");
    // for (int row = 0; row < rows; row++)
    // {
    //     for (int col = 0; col < cols; col++)
    //     {
    //         printf("%6.1f ", h_B[row * cols +col]);
    //     }
    //     printf("\n");
        
    // }

    float *d_A;
    float *d_B;
    float *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(
        d_A, h_A, size, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, size, cudaMemcpyHostToDevice
    ));

    dim3 threadsPerBlock(2,2);

    dim3 blocksPerGrid(
        (cols+threadsPerBlock.x-1) / threadsPerBlock.x,
        (rows+threadsPerBlock.y-1) / threadsPerBlock.y
    );

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C, rows, cols);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C,d_C, size, cudaMemcpyDeviceToHost));

    printf("h_C \n");
    // for (int row = 0; row < rows; row++)
    // {
    //     for (int col = 0; col < cols; col++)
    //     {
    //         printf("%6.1f ", h_C[row * cols +col]);
    //     }
    //     printf("\n");
        
    // }
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));


    return 0;
}