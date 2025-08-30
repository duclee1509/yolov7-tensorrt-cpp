Check out my blog on how to build and run your custom model.

[🚀 C++ Object Detection Yolov7 Tensorrt](https://www.duclee.com/posts/yolov7-tensorrt-cpp)

# Simple Usage
### 1. Make sure your environment has NVIDIA GPU and the following drivers:
* NVIDIA Driver and CUDA Toolkit - [🚀 NVIDIA Driver and CUDA Installation on Ubuntu](https://www.duclee.com/posts/nvidia-cuda-toolkit-installation/)
* OpenCV C++ - [🚀 OpenCV CUDA Installation for C++ Application on Ubuntu](https://www.duclee.com/posts/opencv-c++-gpu-installation)
* Other stubs like cmake, gcc/g++, ...

### 2. Build the source code
```
mkdir build
cd build
cmake ..
make
```

### 3. Detect
```
cd build
./Yolov7_Object_Detection
```

**Output**: `output.jpg`

![](/output.jpg)
