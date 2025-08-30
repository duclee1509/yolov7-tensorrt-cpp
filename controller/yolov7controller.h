#ifndef YOLOV7CONTROLLER_H
#define YOLOV7CONTROLLER_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <string.h>
#include <iostream>
#include "config.h"
#include "modelAPI.h"

using namespace cv;
using namespace std;
using namespace nvinfer1;

class Component
{
public:
    Component();
    ~Component();

    void read_model(string model_name);
    void create_model();
    void setup(string model_name);

    void detect(cv::Mat &image,
                std::vector<cv::Rect> &boxes,
                std::vector<int> &classIds,
                std::vector<float> &scores);

private:
    size_t size{0};
    ICudaEngine *engine;
    IRuntime *runtime;
    IExecutionContext *context;
    cudaStream_t stream;
    std::vector<std::string> classes;
    float* device_buffers[2];
    float* output_buffer_host;
};

class ObjectDetection
{
public:
    ObjectDetection(std::string model_path, std::string class_path);
    ~ObjectDetection();

    Mat detect(cv::Mat image);

private:
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> scores;

    Component *pComponent = nullptr;
};

#endif // YOLOV7CONTROLLER_H
