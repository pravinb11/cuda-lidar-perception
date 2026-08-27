# Real-Time GPU LiDAR Processing and 3D Perception using CUDA

A GPU-accelerated LiDAR processing and 3D perception pipeline developed using NVIDIA CUDA and c++

## Topics completed
### 1. Hello World
### 2. Vector add
### 3. CUDA Memory management
### 4. CUDA error handling -- ToDo


## 🔨 Build Instructions

### 1. Clone the repository

```bash
git clone https://github.com/pravinb11/cuda-lidar-perception.git
cd cuda_lidar_project
```

### 2. Configure the project

Generate the build files using CMake:

```bash
cmake -S . -B build
```

### 3. Build the project

Compile the CUDA/C++ source files:

```bash
cmake --build build -j
```

### 4. Run the program

For example:

```bash
./build/vector_add
```

### 5. Clean and rebuild

If a completely fresh build is required:

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j
```

### Build Workflow

```text
Source Code
     ↓
CMakeLists.txt
     ↓
cmake -S . -B build
     ↓
Build Configuration
     ↓
cmake --build build -j
     ↓
Executable
     ↓
./build/<program>
```
