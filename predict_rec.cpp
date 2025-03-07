
#include "predict_rec.h"

TextRecognizer::TextRecognizer(const std::string& recModelPath, const std::string& char_dict_path, bool useGpu) {
    // 初始化 ONNX Runtime 环境
    this->getSession(recModelPath, useGpu, "TextRecognizer");
    this->recInputName = this->getInputName(this->recOnnxSession);
    this->recOutputName = this->getOutputName(this->recOnnxSession);
    this->postProcessOp = CTCLabelDecode(char_dict_path, true);

}
void TextRecognizer::getSession(const std::string& modelPath, bool isGpu, const std::string& modelType){
    this->env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, modelType.c_str());
    this->sessionOptions = Ort::SessionOptions();
//        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
//    OrtCUDAProviderOptions cudaOption;
    OrtTensorRTProviderOptions trt_options{};
    trt_options.device_id = 0;
    trt_options.trt_max_workspace_size = 2147483648;
    trt_options.trt_max_partition_iterations = 10;
    trt_options.trt_min_subgraph_size = 5;
    trt_options.trt_fp16_enable = 1;
    trt_options.trt_int8_use_native_calibration_table = 1;
    trt_options.trt_engine_cache_enable = 1;
    trt_options.trt_engine_cache_path = "./cache_trt/";
    trt_options.trt_dump_subgraphs = 1;
    if (isGpu && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGpu && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        this->sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
//        this->sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
        std::cout << modelPath << std::endl;
    }
    this->recOnnxSession = Ort::Session(this->env, std::wstring(modelPath.begin(), modelPath.end()).c_str(), this->sessionOptions);
}
std::vector<std::pair<std::string, float>> TextRecognizer::operator()(const std::vector<cv::Mat>& img_list) {
    std::vector<const char*> inputNamesPtr;
    for (const std::string& name : this->recInputName) {
        inputNamesPtr.push_back(name.c_str());
    }
    std::vector<const char*> outputNamesPtr;
    for (const std::string& name : this->recOutputName) {
        outputNamesPtr.push_back(name.c_str());
    }
    std::vector<std::pair<std::string, float>> rec_res(img_list.size(), {"", 0.0f});
    int batch_num = this->rec_batch_num;

    // 计算所有图像的宽高比
    std::vector<float> width_list;
    for (const auto& img : img_list) {
        width_list.push_back(static_cast<float>(img.cols) / img.rows);
    }
    // 对图像按宽高比排序
    std::vector<size_t> indices(img_list.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&width_list](size_t i1, size_t i2) {
        return width_list[i1] < width_list[i2];
    });
    // 分批处理图像
    for (size_t beg_img_no = 0; beg_img_no < img_list.size(); beg_img_no += batch_num) {
        size_t end_img_no = std::min(img_list.size(), beg_img_no + batch_num);
        std::vector<cv::Mat> norm_img_batch;

        // 计算当前批次的最大宽高比
        float max_wh_ratio = 0;
        for (size_t i = beg_img_no; i < end_img_no; ++i) {
            float wh_ratio = width_list[indices[i]];
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        // 对当前批次的图像进行归一化处理
        for (size_t i = beg_img_no; i < end_img_no; ++i) {
            cv::Mat norm_img = resize_norm_img(img_list[indices[i]], max_wh_ratio, this->rec_image_shape);
            norm_img_batch.push_back(norm_img);
        }

        // 将图像数据转换为 ONNX 输入格式
        std::vector<float> blob = create_input_tensor(norm_img_batch);
        std::vector<int64_t> input_shape = {static_cast<int64_t>(norm_img_batch.size()), 3, 48, static_cast<int64_t>(48*max_wh_ratio)};
        size_t input_tensor_size = norm_img_batch.size() * 3 * 48 * input_shape[3];
        std::vector<Ort::Value> inputTensors;
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, blob.data(), input_tensor_size,
                input_shape.data(), input_shape.size()));
        // 运行模型推理
        std::vector<Ort::Value> output_tensors = this->recOnnxSession.Run(Ort::RunOptions{nullptr},
                                                                          inputNamesPtr.data(),
                                                                          inputTensors.data(),
                                                                          inputNamesPtr.size(),
                                                                          outputNamesPtr.data(),
                                                                          outputNamesPtr.size());
        std::vector<std::pair<std::string, float>> res = this->postProcessOp(output_tensors);
        for (size_t i = 0; i < res.size(); ++i) {
            rec_res[indices[beg_img_no + i]] = res[i];
        }

    }

    return rec_res;
}


cv::Mat TextRecognizer::resize_norm_img(const cv::Mat& img, float max_wh_ratio, const std::vector<int>& rec_image_shape) {
    int imgC = rec_image_shape[0];
    int imgH = rec_image_shape[1];
    int imgW = static_cast<int>(imgH * max_wh_ratio);

    // 确保输入图像的通道数匹配
    CV_Assert(imgC == img.channels());

    int h = img.rows;
    int w = img.cols;
    float ratio = static_cast<float>(w) / h;
    int resized_w = (std::ceil(imgH * ratio) > imgW) ? imgW : static_cast<int>(std::ceil(imgH * ratio));

    // 调整图像大小
    cv::Mat resized_image;
    cv::resize(img, resized_image, cv::Size(resized_w, imgH));

    // 转换为 float32 类型
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255.0);

    // 归一化 (减 0.5 再除以 0.5)
//        cv::subtract(resized_image, cv::Scalar(0.5), resized_image);
//        cv::divide(resized_image, cv::Scalar(0.5), resized_image);
    resized_image = (resized_image - 0.5) / 0.5;
    // 创建填充后的图像
    cv::Mat padding_im = cv::Mat::zeros(cv::Size(imgW, imgH), CV_32FC3);
    resized_image.copyTo(padding_im(cv::Rect(0, 0, resized_w, imgH)));

    return padding_im;
}

// 创建 ONNX 输入张量
std::vector<float> TextRecognizer::create_input_tensor(const std::vector<cv::Mat>& img_batch) {
    std::vector<float> input_tensor_values;
    for (int i = 0; i < img_batch.size(); i++){
        // HWC -> CHW 格式
        std::vector<cv::Mat> channels(3);
        split(img_batch[i], channels);
        for (auto &ch : channels) {

            input_tensor_values.insert(input_tensor_values.end(), (float *)ch.datastart, (float *)ch.dataend);
        }
    }
    return input_tensor_values;
}



