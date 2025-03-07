//
// Created by admin on 2025/2/26.
//
#ifndef DB_POSTPROCESS_H
#define DB_POSTPROCESS_H
#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "clipper.hpp"

class DBPostProcess {
public:
    DBPostProcess() = default;
    std::vector<std::vector<cv::Point>> boxes_from_bitmap(const cv::Mat &pred, const cv::Mat &bitmap, int dest_width, int dest_height);

    std::vector<cv::Point> unclip(const std::vector<cv::Point> &box, float unclip_ratio);

    std::pair<std::vector<cv::Point>, float> get_mini_boxes(const std::vector<cv::Point> &contour);

    float box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point> &box);

    float box_score_slow(const cv::Mat &bitmap, const std::vector<cv::Point> &contour);

    std::vector<std::vector<cv::Point>>
    operator()(const cv::Mat &preds, const std::tuple<int, int, float, float> &shape_list);

private:
    float thresh = 0.3;
    float box_thresh = 0.6;
    int max_candidates = 1000;
    float unclip_ratio = 1.5;
    int min_size = 3;
    std::string score_mode = "fast";
    std::string box_type = "quad";
    cv::Mat dilation_kernel;
};
#endif //DB_POSTPROCESS_H