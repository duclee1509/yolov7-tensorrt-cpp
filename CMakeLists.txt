cmake_minimum_required(VERSION 3.10)

project(Yolov7_Object_Detection)

# Compiler and language options
add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

# CUDA configuration
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/yolov7/include)
include_directories(${PROJECT_SOURCE_DIR}/yolov7/plugin)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
    include_directories(${PROJECT_SOURCE_DIR}/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/include)
    link_directories(${PROJECT_SOURCE_DIR}/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib)
endif()

add_library(preprocess SHARED ${PROJECT_SOURCE_DIR}/yolov7/src/preprocess.cu)
set_target_properties(preprocess PROPERTIES CUDA_ARCHITECTURES 86)
add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yolov7/plugin/yololayer.cu)
set_target_properties(myplugins PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(myplugins preprocess nvinfer cudart)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# OpenCV configuration
include_directories(${PROJECT_SOURCE_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/yolov7/src/*.cpp ${PROJECT_SOURCE_DIR}/yolov7/src/*.cu)

# NCNN library configuration
include_directories(/usr/local/include/ncnn)
link_directories(/usr/local/lib/ncnn)
include_directories(/usr/include)
link_directories(usr/lib/aarch64-linux-gnu)

# Source files declaration
set(PROJECT_SOURCES
    yolov7/src/block.cpp
    yolov7/src/modelAPI.cpp
    yolov7/src/postprocess.cpp

    yolov7/include/block.h
    yolov7/include/config.h
    yolov7/include/cuda_utils.h
    yolov7/include/logging.h
    yolov7/include/preprocess.h
    yolov7/include/types.h
    yolov7/include/utils.h
    yolov7/include/macros.h
    yolov7/include/modelAPI.h
    yolov7/include/postprocess.h
    yolov7/plugin/yololayer.h

    controller/yolov7controller.cpp
    controller/yolov7controller.h

    main.cpp
)

# Define executable target
add_executable(Yolov7_Object_Detection ${PROJECT_SOURCES})
set_target_properties(Yolov7_Object_Detection PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(Yolov7_Object_Detection PUBLIC
    ${OpenCV_LIBS}
    nvinfer
    preprocess
    cudart
    myplugins)
target_include_directories(Yolov7_Object_Detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_compile_definitions(Yolov7_Object_Detection PRIVATE
    CONSOLE
    WARN_ON
)
