#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A,
                          const float *B,
                          float *C,
                          int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 5;
    // const int N = 1000000;
    const int size = N * sizeof(float);

    float h_A[N] = {10, 20, 30, 40, 50};
    float h_B[N] = {1, 2, 3, 4, 5};
    float h_C[N];

    // float *h_A = new float[N];
    // float *h_B = new float[N];
    // float *h_C = new float[N];
    // for (int i = 0; i < N; i++)
    // {
    //     h_A[i] = i;
    //     h_B[i] = 2 * i;
    // }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);


    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("C[%d] = %.1f\n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}