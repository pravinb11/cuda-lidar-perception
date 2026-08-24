cmake_minimum_required(VERSION 3.18)

project(cuda_lidar
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(hello_cuda
    src/hello_cuda.cu
)

set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)