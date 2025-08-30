#include "yolov7controller.h"

ObjectDetection::ObjectDetection(std::string model_path, std::string class_path)
{
    pComponent = new Component();
    pComponent->create_model();
    pComponent->setup(class_path);
    pComponent->read_model(model_path);

    // Run a dummy inference to initialize TensorRT buffers and CUDA stream
    cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
    pComponent->detect(dummy, boxes, classIds, scores);
}

ObjectDetection::~ObjectDetection()
{
    delete pComponent;
}

Mat ObjectDetection::detect(Mat image)
{
    auto start = std::chrono::high_resolution_clock::now();

    pComponent->detect(image, boxes, classIds, scores);

    auto end = std::chrono::high_resolution_clock::now();

    double time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double fps = 1000.0/time_elapsed;
    std::cout << "Time elapsed: " << time_elapsed << " ms" << std::endl;
    std::cout << "FPS: " << fps << std::endl;

    return image;
}

Component::Component()
{
    cudaSetDevice(kGpuId);
}

Component::~Component()
{
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void Component::detect(cv::Mat &image,
                        std::vector<cv::Rect> &boxes,
                        std::vector<int> &classIds,
                        std::vector<float> &scores)
{
    auto start = std::chrono::high_resolution_clock::now();
    double time_elapsed;

    boxes.clear();
    classIds.clear();
    scores.clear();

    // Preprocess
    std::vector<cv::Mat> batch_imgs{image};
    cuda_batch_preprocess(batch_imgs, device_buffers[0], kInputW, kInputH, stream);
    // Run inference
    infer(*context, stream, (void**)device_buffers, output_buffer_host, kBatchSize);
    // NMS
    std::vector<std::vector<Detection>> res_batch;
    batch_nms(res_batch, output_buffer_host, 1, kOutputSize, kConfThresh, kNmsThresh);
    // Draw bounding boxes
    auto& res = res_batch[0];
    for (size_t j = 0; j < res.size(); j++) {
        cv::Rect r = get_rect(image, res[j].bbox);
        boxes.push_back(r);
        classIds.push_back((int)res[j].class_id);
        scores.push_back(res[j].conf);
        cv::rectangle(image, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(image, classes[classIds[j]], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}

void Component::read_model(string model_name)
{
    deserialize_engine(model_name, &runtime, &engine, &context);
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host);
}

void Component::create_model()
{
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
}

void Component::setup(string model_name)
{
    // Get class name for lp detection
    std::ifstream file_classes(model_name);
    std::string line;
    if (file_classes.is_open())
    {
        while (std::getline(file_classes, line))
        {
            classes.push_back(line);
        }
        file_classes.close();
    }
    else
    {
        std::cout << "Can not open file " << model_name << std::endl;
    }
}
