#include <stdio.h>
#include <cuda_runtime.h>

__global__ void multiplyByTwo(float *data, int N)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    if (i < N)
    {
        data[i] = data[i] * 2.0f;
    }
    
}
int main()
{
    const int N = 5;
    const int size = N * sizeof(float);

    // Host memory

    float h_data[N] = {10, 20, 30, 40, 50};

    printf("Host data before copy: \n");

    for (int i = 0; i < N; i++)
    {
        printf("h_data[%d] = %0.1f \n", i, h_data[i]);
    }

    // Device memory

    float *d_data;
    cudaMalloc(&d_data, size);

    // Host to Device

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    multiplyByTwo<<<1,2>>>(d_data, N);
    // Device to host

    float h_result[N];

    cudaMemcpy(h_result,d_data,size,cudaMemcpyDeviceToHost);

    printf("\nData after GPU round trip: \n");

    for (int i = 0; i < N; i++)
    {
        printf("h_result[%d] = %0.1f \n", i, h_result[i]);
    }

    // Free device memory
    cudaFree(d_data);
    

    
    return 0;
}