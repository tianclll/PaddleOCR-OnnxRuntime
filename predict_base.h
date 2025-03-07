//
// Created by admin on 2025/2/25.
//
#ifndef PREDICT_BASE_H
#define PREDICT_BASE_H
#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <algorithm>
#include <xstring>
#include <opencv2/opencv.hpp>
class PredictBase {
public:
    PredictBase() = default;
    std::vector<std::string> getInputName(Ort::Session& session);
    std::vector<std::string> getOutputName(Ort::Session& session);
};
#endif // PREDICT_BASE_H