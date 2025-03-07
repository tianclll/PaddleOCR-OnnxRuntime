//
// Created by admin on 2025/2/25.
//
#ifndef PREDICT_DET_H
#define PREDICT_DET_H
#include "predict_base.h"
#include <iostream>
#include "operators.h"
#include "utils.h"
#include "onnxruntime_cxx_api.h"
#include "db_postprocess.h"
class TextDetector : public PredictBase {
public:
    explicit TextDetector(const std::string& detModelPath, bool useGpu = true);
    std::vector<std::vector<cv::Point>> operator()(cv::Mat& img);
    void getSession(const std::string& modelPath, bool isGpu, const std::string& modelType);

protected:
    Ort::Env env {nullptr};
    Ort::SessionOptions sessionOptions;
    Ort::Session detOnnxSession {nullptr};
    DBPostProcess postprocessOp;
    std::vector<std::string> detInputName;
    std::vector<std::string> detOutputName;

    PredictData predictData;
    NormalizeImage normalizeImage;
    DetResizeForTest detResizeForTest;
    ToCHWImage toCHWImage;
};
#endif //PREDICT_DET_H