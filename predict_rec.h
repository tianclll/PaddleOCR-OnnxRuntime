//
// Created by admin on 2025/3/4.
//
#ifndef PREDICT_REC_H
#define PREDICT_REC_H
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "predict_base.h"
#include "rec_postprocess.h"
class TextRecognizer: public PredictBase {
public:
    TextRecognizer(const std::string& recModelPath, const std::string& char_dict_path, bool useGpu);
    void getSession(const std::string& modelPath, bool isGpu, const std::string& modelType);
    std::vector<std::pair<std::string, float>> operator()(const std::vector<cv::Mat>& img_list);
    cv::Mat resize_norm_img(const cv::Mat& img, float max_wh_ratio, const std::vector<int>& rec_image_shape);
    std::vector<float> create_input_tensor(const std::vector<cv::Mat>& img_batch);
private:
    Ort::Env env {nullptr};
    Ort::SessionOptions sessionOptions;
    Ort::Session recOnnxSession {nullptr};
    std::vector<std::string> recInputName;
    std::vector<std::string> recOutputName;
    int rec_batch_num = 6;
    std::vector<int> rec_image_shape {3, 48, 320};
    CTCLabelDecode postProcessOp;
};
#endif //PREDICT_REC_H