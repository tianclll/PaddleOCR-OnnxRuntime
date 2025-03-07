#ifndef OPERATORS_H
#define OPERATORS_H
#include <opencv2/core.hpp>
//#include "NumCpp.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
struct PredictData{
    cv::Mat Image;
    std::tuple<int, int, float, float> Shape;
};
class NormalizeImage {
public:
    NormalizeImage(float scale = 1.0 / 255.0,
                   std::vector<float> mean = {0.485, 0.456, 0.406},
                   std::vector<float> std = {0.229, 0.224, 0.225},
                   const std::string& order = "hwc");
    cv::Mat operator()(const cv::Mat& img);
private:
    float scale;
    std::vector<float> mean;
    std::vector<float> std;
    std::vector<int> meanShape;
    std::vector<int> stdShape;
};

class DetResizeForTest {
public:
    explicit DetResizeForTest(int det_limit_side_len = 960, const std::string& det_limit_type = "max");
    PredictData operator()(cv::Mat& img);
private:
    int limit_side_len;
    std::vector<int> image_shape;
    std::string limit_type;
    static cv::Mat image_padding(const cv::Mat& img);
    std::tuple<cv::Mat, float, float> resize_image_type0(const cv::Mat& img);
};

class ToCHWImage {
public:
    ToCHWImage() = default;
    PredictData operator()(PredictData& data, float*& blob);
};
#endif // OPERATORS_H