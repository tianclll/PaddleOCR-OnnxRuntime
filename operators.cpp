//
// Created by admin on 2025/2/24.
//

#include "operators.h"

NormalizeImage::NormalizeImage(float scale,std::vector<float> mean,std::vector<float> std,const std::string& order) {
    // Initialize scale
    this->scale = scale;

    // Initialize mean and std
    this->mean = std::move(mean);
    this->std = std::move(std);

    // Shape for reshaping based on order
    if (order == "chw") {
        // 'chw' means 3 channels in the first dimension (Channel, Height, Width)
        this->meanShape = {3, 1, 1};
        this->stdShape = {3, 1, 1};
    } else {
        // 'hwc' means Height, Width, Channel (common in OpenCV)
        this->meanShape = {1, 1, 3};
        this->stdShape = {1, 1, 3};
    }
}

cv::Mat NormalizeImage::operator()(const cv::Mat& img) {
        // Ensure the image is in float32
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F);

        // Apply the scale factor
        img_float = img_float * this->scale;
        // Normalize with mean and std
        cv::Mat result = img_float.clone();

        for (int c = 0; c < 3; ++c) {
            // Process each channel separately
            for (int row = 0; row < img_float.rows; ++row) {
                for (int col = 0; col < img_float.cols; ++col) {
                    // Normalize pixel value for each channel
                    result.at<cv::Vec3f>(row, col)[c] =
                            (img_float.at<cv::Vec3f>(row, col)[c] - this->mean[c]) / this->std[c];
                }
            }
        }

        return result;
}

DetResizeForTest::DetResizeForTest(int det_limit_side_len, const std::string& det_limit_type) {

    this->limit_side_len = det_limit_side_len;
    this->limit_type = det_limit_type;
}
PredictData DetResizeForTest::operator()(cv::Mat& img) {
    PredictData data;
    int src_h = img.rows;
    int src_w = img.cols;
    if (src_h + src_w < 64) {
        img = image_padding(img);
    }

    float ratio_h = 1.0, ratio_w = 1.0;
    cv::Mat resized_img;
    std::tie(resized_img, ratio_h, ratio_w) = resize_image_type0(img);
    std::tuple<int, int, float, float> shape = std::make_tuple(src_h, src_w, ratio_h, ratio_w);
    data.Image = resized_img;
    data.Shape = shape;
    return data;
}

// Padding method
 cv::Mat DetResizeForTest::image_padding(const cv::Mat& img) {
    int h = img.rows, w = img.cols, c = img.channels();
    cv::Mat padded_img = cv::Mat::zeros(std::max(32, h), std::max(32, w), img.type());
    img.copyTo(padded_img(cv::Rect(0, 0, w, h)));
    return padded_img;
}

// Resize type 0
std::tuple<cv::Mat, float, float> DetResizeForTest::resize_image_type0(const cv::Mat& img) {
    float ratio = 1.0;
    int h = img.rows, w = img.cols;
    // Apply limit based on side length
    if (std::max(h, w) > this->limit_side_len) {
        if (h > w) {
            ratio = (float)this->limit_side_len / (float)h;
        }
        else {
            ratio = (float)this->limit_side_len / (float)w;
        }
    }
    else{
        ratio = 1.0f;
//        ratio = (float)this->limit_side_len / (float)w;
    }

    int resize_h = int(h * ratio);
    int resize_w = int(w * ratio);

    resize_h = std::max(int(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(int(std::round(resize_w / 32.0) * 32), 32);
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resize_w, resize_h));
    float ratio_h = (float)resize_h / h;
    float ratio_w = (float)resize_w / w;
    return std::make_tuple(resized_img, ratio_h, ratio_w);
}


// 将 HWC 图像转换为 CHW 格式
PredictData ToCHWImage::operator()(PredictData& data, float*& blob) {
    cv::Mat img = data.Image;
//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // 检查输入图像是否为空
    if (img.empty()) {
        std::cerr << "Input image is empty!" << std::endl;
        return data;
    }

    // 如果输入图像是 3 通道 HWC 格式，转换为 CHW 格式
    if (img.channels() == 3) {
        // 获取图像的尺寸
        int height = img.rows;
        int width = img.cols;

        // 为 (C, H, W) 格式的图像分配内存
        blob = new float[height * width * img.channels()]; // CxHxW 格式的内存

        // 维度变换：从 (H, W, C) 转为 (C, H, W)
        std::vector<cv::Mat> channels(3);
        cv::split(img, channels); // 将图像分成3个通道

        // 处理每个通道
        for (int c = 0; c < 3; ++c) {
            // 为每个通道分配相应的内存区域
            cv::Mat channel(height, width, CV_32FC1, blob + c * height * width); // 每个通道是一个矩阵
            // 将每个通道的像素值转换到 [0, 1] 范围并转换为浮点型
            channels[c].convertTo(channel, CV_32F); // 归一化到 [0,1]
        }

        // 显示图像
        cv::Mat display_img;
        cv::merge(channels, display_img);
//        cv::normalize(display_img, display_img, 0, 255, cv::NORM_MINMAX);
        display_img.convertTo(display_img, CV_8UC3);

        // 更新数据中的图像
        data.Image = display_img;



    } else {
        std::cout << "Input image is not 3 channels. Skipping CHW conversion." << std::endl;
    }

    return data;
}

