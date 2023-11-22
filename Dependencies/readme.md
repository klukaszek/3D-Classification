# Dependencies Installation Guide

## Trimesh 2 Fork - Jeroen Baert (https://github.com/Forceflow/trimesh2)

### Requirements

- mesa-utils
- freeglut3-dev

### Installation

Navigate to the trimesh2 folder in Dependencies and run the following commands:

Ubuntu/Debian:
```bash
sudo apt-get install mesa-utils freeglut3-dev
make
```

Windows:
```
For Windows, build solutions for VS2022 and VS2019 are provided in the mscvfolder, verified working with the free Community Editions of Visual Studio. The solutions contain both Debug and Release profiles for 32-bit and 64-bit builds.

The built libraries will be placed in a folder named lib.(architecture).(visual studio version) in the trimesh2 root folder. For example, for a 64-bit Visual Studio 2017 build, it will be lib.win64.vs141. 

The utilities will be placed in util.(architecture).(visual studio version). This naming scheme is in place to avoid clashing trimesh2 versions.
```

OSX:
```
make
```

## Cuda Voxelizer - Jeroen Baert (https://github.com/Forceflow/cuda_voxelizer)

### Requirements

- NVIDIA CUDA Toolkit 10.0 or higher
- Trimesh2 (see above)
- OpenMP

### Installation

Navigate to the cuda_voxelizer folder in Dependencies and run the following commands:

Ubuntu/Debian:

```bash
sudo apt-get install nvidia-cuda-toolkit

cd build
cmake CUDAARCHS="your_cuda_compute_capability" cmake -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" -DCUDA_ARCH:STRING="your_cuda_compute_capability" ..
cmake --build . --parallel num_cores
```

Windows:
- Install CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
```bash
$env:CUDAARCHS="your_cuda_compute_capability"
cmake -A x64 -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" ..
cmake --build . --parallel num_cores
```

OSX:
NOTE: Modern OSX does not support CUDA, so this will not work. You can try to build the CPU version of the voxelizer, it is slower, but still faster than the original Python implementation.
```bash

cd build
cmake -DTrimesh2_INCLUDE_DIR:PATH="path_to_trimesh2_include" -DTrimesh2_LINK_DIR:PATH="path_to_trimesh2_library_dir" ..
cmake --build . --parallel num_cores
```
Note: CUDAARCHS should just be the compute capability, e.g. 75 for RTX 20 series cards, and 86 for RTX 30 series cards. For a full list, see https://developer.nvidia.com/cuda-gpus (omit the dot in the compute capability).
