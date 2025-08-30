#ifndef _MODEL_API_H_
#define _MODEL_API_H_

#include "config.h"
#include "cuda_utils.h"
#include "logging.h"
#include "utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "yololayer.h"
#include <chrono>
#include <fstream>

using namespace nvinfer1;

const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
static Logger gLogger;

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine, IExecutionContext** context);
void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device, float** output_buffer_host);
void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchSize);
bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, std::string& img_dir, std::string& sub_type);

#endif
