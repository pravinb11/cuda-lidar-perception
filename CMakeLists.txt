cmake_minimum_required(VERSION 3.18)

project(cuda_lidar
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(hello_cuda
    src/hello_cuda.cu
)

add_executable(vector_add
    src/vector_add.cu
)

add_executable(memory_demo
    src/memory_demo.cu
)

add_executable(error_demo
    src/error_demo.cu
)

add_executable(cuda_timing
    src/cuda_timing.cu
)

add_executable(matrix_add
    src/matrix_add.cu
)

target_include_directories(error_demo PRIVATE
    include
)

target_include_directories(cuda_timing PRIVATE
    include
)
target_include_directories(matrix_add PRIVATE
    include
)

add_executable(memory_access
    src/memory_access.cu
)

target_include_directories(memory_access PRIVATE
    include
)

add_executable(shared_reduction
    src/shared_reduction.cu
)

target_include_directories(shared_reduction PRIVATE
    include
)

add_executable(matrix_multiply
    src/matrix_multiply.cu
)

target_include_directories(matrix_multiply PRIVATE
    include
)

add_executable(reduction
    src/reduction.cu
)

target_include_directories(reduction PRIVATE
    include
)

add_executable(warp_demo
    src/warp_demo.cu
)

target_include_directories(warp_demo PRIVATE
    include
)

add_executable(scan
    src/scan.cu
)

target_include_directories(scan PRIVATE
    include
)


add_executable(compact
    src/compact.cu
)

target_include_directories(compact PRIVATE
    include
)

add_executable(lidar_filter
    src/lidar_filter.cu
)

target_include_directories(lidar_filter PRIVATE
    include
)


add_executable(test_loader
    src/test_loader.cpp
    src/pointcloud_loader.cpp
)

target_include_directories(test_loader PRIVATE
    include
)


add_executable(lidar_filter_pcd
    src/lidar_filter_pcd.cu
    src/pointcloud_loader.cpp
)

target_include_directories(lidar_filter_pcd PRIVATE
    include
)

set_target_properties(hello_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)