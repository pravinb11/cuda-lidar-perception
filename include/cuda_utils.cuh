#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t error = call;                                \
                                                              \
    if (error != cudaSuccess)                                \
    {                                                         \
        fprintf(stderr,                                       \
                "CUDA Error: %s:%d\n%s\n",                   \
                __FILE__,                                    \
                __LINE__,                                    \
                cudaGetErrorString(error));                  \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)