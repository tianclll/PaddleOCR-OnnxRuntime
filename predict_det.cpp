//
// Created by admin on 2025/2/25.
//
#include "predict_det.h"

TextDetector::TextDetector(const std::string& detModelPath, bool useGpu){
    // 前处理
    float scale = 1.0/255.0;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};
    const std::string& order = "hwc";
    this->normalizeImage = NormalizeImage(scale, mean, std, order);
    this->detResizeForTest = DetResizeForTest(960, "max");
    this->toCHWImage = ToCHWImage();
    this->postprocessOp = DBPostProcess();
    this->getSession(detModelPath, useGpu, "TextDetector");
    this->detInputName = this->getInputName(this->detOnnxSession);
    this->detOutputName = this->getOutputName(this->detOnnxSession);

}
void TextDetector::getSession(const std::string& modelPath, bool isGpu, const std::string& modelType){
    this->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, modelType.c_str());
    this->sessionOptions = Ort::SessionOptions();
//        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    OrtTensorRTProviderOptions trtOptions{};
    trtOptions.device_id = 0;
    trtOptions.trt_max_workspace_size = 2147483648;
    trtOptions.trt_max_partition_iterations = 10;
    trtOptions.trt_min_subgraph_size = 5;
    trtOptions.trt_fp16_enable = 1;
    trtOptions.trt_int8_use_native_calibration_table = 1;
    trtOptions.trt_engine_cache_enable = 1;
    trtOptions.trt_engine_cache_path = "./cache_trt/";
    trtOptions.trt_dump_subgraphs = 1;
    if (isGpu && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGpu && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        this->sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
//        this->sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
        std::cout << modelPath << std::endl;
    }
    this->detOnnxSession = Ort::Session(this->env, std::wstring(modelPath.begin(), modelPath.end()).c_str(), this->sessionOptions);
}
std::vector<std::vector<cv::Point>> TextDetector::operator()(cv::Mat& img) {
    cv::Mat image = img.clone();
    int w = image.cols;
    int h = image.rows;
    // 3. 将 std::vector<std::string> 转换为 std::vector<const char*>
    std::vector<const char*> inputNamesPtr;
    for (const std::string& name : this->detInputName) {
        inputNamesPtr.push_back(name.c_str());
    }

    std::vector<const char*> outputNamesPtr;
    for (const std::string& name : this->detOutputName) {
        outputNamesPtr.push_back(name.c_str());
    }
    this->predictData = this->detResizeForTest(image);
    cv::Mat normalize_image = this->normalizeImage(this->predictData.Image);
    this->predictData.Image = normalize_image;
    float* blob = nullptr;
    this->predictData = this->toCHWImage(this->predictData, blob);
    cv::Mat predict_image = this->predictData.Image;
    std::vector<int64_t> input_shape = {1, 3, predict_image.rows, predict_image.cols};
    // 5. 将图像数据转换为 ONNX Tensor
//    matToVector(predict_image, blob);  // 转为 1D 数组
    size_t inputTensorSize = vectorProduct(input_shape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            input_shape.data(), input_shape.size()
    ));

    std::vector<Ort::Value> output_tensors = this->detOnnxSession.Run(Ort::RunOptions{nullptr},
                                                                      inputNamesPtr.data(),
                                                                      inputTensors.data(),
                                                                      inputNamesPtr.size(),
                                                                      outputNamesPtr.data(),
                                                                      outputNamesPtr.size());


    const float* floatArray = output_tensors[0].GetTensorMutableData<float>();
    int outputCount = 1;
    for(int i=0; i < output_tensors.at(0).GetTensorTypeAndShapeInfo().GetShape().size(); i++)
    {
        int dim = output_tensors.at(0).GetTensorTypeAndShapeInfo().GetShape().at(i);
        outputCount *= dim;
    }
    cv::Mat binary(predict_image.rows, predict_image.cols, CV_32FC1);
    memcpy(binary.data, floatArray, outputCount * sizeof(float));
    std::vector<std::vector<cv::Point>> dt_boxes = this->postprocessOp(binary, predictData.Shape);

    delete[] blob;
    return dt_boxes;
}
