#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"
// A : MxK B: KxN
// To access of row 5*K+1, col 2*N+1
// A: [a,b;c,d] B: [e,f;g,h] C[0,0]:[a*e+b*g]
__global__ void matrixMultiplyNaive(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < K && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            sum += A[row*K+k] * B[k*N+col];
        }

        C[row*N+col] = sum;  // row *N goes to that row then +col choose col element
        
    }
}


#define TILE_SIZE 16

__global__ void matrixMultiplyTiled(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++)
    {
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < K)
        {
            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        if (bRow < K && col < N)
        {
            tileB[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] =
                0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main()
{
    const int M = 1000;
    const int K = 1000;
    const int N = 1000;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
   
    // float h_A[M * K];
    // float h_B[K * N];
    // float h_C[M * N];
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++)
    {
        h_A[i] = i+1.0f;
    }

    for (int i = 0; i < K * N; i++)
    {
        h_B[i] = 2.0f;
    }

    printf("Matrix A:\n");

    // for (int row = 0; row < M; row++)
    // {
    //     for (int col = 0; col < N; col++)
    //     {
    //         printf("%6.1f ",
    //                h_A[row * N + col]);
    //     }

    //     printf("\n");
    // }

    // printf("Matrix B:\n");

    // for (int row = 0; row < M; row++)
    // {
    //     for (int col = 0; col < N; col++)
    //     {
    //         printf("%6.1f ",
    //                h_B[row * N + col]);
    //     }

    //     printf("\n");
    // }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));


    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        sizeA,
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        sizeB,
        cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1)
        / threadsPerBlock.x,

        (M + threadsPerBlock.y - 1)
        / threadsPerBlock.y
    );

    // matrixMultiplyNaive<<<
    //     blocksPerGrid,
    //     threadsPerBlock
    // >>>(
    //     d_A,
    //     d_B,
    //     d_C,
    //     M,
    //     K,
    //     N
    // );

    matrixMultiplyTiled<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        sizeC,
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    printf(
        "Kernel time: %.3f ms\n",
        milliseconds
    );

    printf("Matrix C:\n");

    // for (int row = 0; row < M; row++)
    // {
    //     for (int col = 0; col < N; col++)
    //     {
    //         printf("%6.1f ",
    //                h_C[row * N + col]);
    //     }

    //     printf("\n");
    // }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0;
}
