//
// Created by admin on 2025/2/24.
//

#include <opencv2/opencv.hpp>
#include <cmath>
cv::Mat get_rotate_crop_image(cv::Mat& img, std::vector<cv::Point2f>& points);
cv::Mat getMinAreaRectCrop(cv::Mat& img, std::vector<cv::Point> cv_points_vec);
size_t vectorProduct(const std::vector<int64_t>& vector);
bool sort_by_top_left(const std::vector<cv::Point>& box1, const std::vector<cv::Point>& box2);
std::vector<std::vector<cv::Point>> sorted_boxes(const std::vector<std::vector<cv::Point>>& dt_boxes);
std::vector<std::vector<int>> convertBoxes(const std::vector<std::vector<cv::Point>>& boxes);