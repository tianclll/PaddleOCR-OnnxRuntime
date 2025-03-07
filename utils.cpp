#include "utils.h"
//cv::Point2f* pointsToCvPoint2f(nc::NdArray<float>& points) {
//    static const int n = points.shape().rows;
//    cv::Point2f* cv_point = new cv::Point2f[n];
//    for (int i = 0; i < points.shape().rows; ++i) {
//        float x = points(i, 0);
//        float y = points(i, 1);
//        cv_point[i] = cv::Point2f(x, y);
//    }
//    return cv_point;
//}
cv::Mat get_rotate_crop_image(cv::Mat& img, std::vector<cv::Point2f>& points) {
    // 计算宽度和高度
    float img_crop_width = std::max(
            cv::norm(points[0] - points[1]),
            cv::norm(points[2] - points[3])
    );
    float img_crop_height = std::max(
            cv::norm(points[0] - points[3]),
            cv::norm(points[1] - points[2])
    );

    // 定义目标图像的四个角点
    cv::Point2f pts_std[4] = { {0, 0},
                               {img_crop_width, 0},
                               {img_crop_width, img_crop_height},
                               {0, img_crop_height} };

    // 计算透视变换矩阵
    cv::Mat M = cv::getPerspectiveTransform(points.data(), pts_std);

    // 应用透视变换
    cv::Mat dst_img;
    cv::warpPerspective(
            img,
            dst_img,
            M,
            cv::Size(static_cast<int>(img_crop_width), static_cast<int>(img_crop_height)),
            cv::BORDER_REPLICATE,
            cv::INTER_CUBIC
    );

    // 如果图像高度大于宽度的1.5倍，则旋转90度
    if (dst_img.rows * 1.0 / dst_img.cols >= 1.5) {
        cv::rotate(dst_img, dst_img, cv::ROTATE_90_CLOCKWISE);
    }

    return dst_img;
}


cv::Mat getMinAreaRectCrop(cv::Mat& img, std::vector<cv::Point> cv_points_vec){
//    cv::Point2f* cv_points = pointsToCvPoint2f(points);
//    std::vector<cv::Point2f> cv_points_vec;
//    for (int i = 0; i < points.shape().rows; ++i) {
//        cv_points_vec.push_back(cv_points[i]);
//    }
//    delete[] cv_points;
    cv::RotatedRect boundingBox = cv::minAreaRect(cv_points_vec);
    cv::Mat boundingBoxPoints;
    cv::boxPoints(boundingBox, boundingBoxPoints);
    std::vector<cv::Point2f> points_vec;

    for (int i = 0; i < boundingBoxPoints.rows; ++i) {
        float x = boundingBoxPoints.at<float>(i, 0);
        float y = boundingBoxPoints.at<float>(i, 1);
        points_vec.push_back(cv::Point2f(x, y));
    }
    // 按照 x 坐标排序
    std::sort(points_vec.begin(), points_vec.end(), [](const cv::Point2f& p1, const cv::Point2f& p2) {
        return p1.x < p2.x;  // 按x坐标排序
    });
    int index_a = 0, index_b = 1, index_c = 2, index_d = 3;
    if (points_vec[1].y > points_vec[0].y) {
        index_a = 0;
        index_d = 1;
    }
    else {
        index_a = 1;
        index_d = 0;
    }
    if (points_vec[3].y > points_vec[2].y) {
        index_b = 2;
        index_c = 3;
    }
    else {
        index_b = 3;
        index_c = 2;
    }
    std::vector<cv::Point2f> box = {{points_vec[index_a].x,points_vec[index_a].y}, {points_vec[index_b].x, points_vec[index_b].y}, {points_vec[index_c].x, points_vec[index_c].y}, {points_vec[index_d].x, points_vec[index_d].y}};
    cv::Mat crop_img = get_rotate_crop_image(img, box);
    return crop_img;
}

size_t vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

bool sort_by_top_left(const std::vector<cv::Point>& box1, const std::vector<cv::Point>& box2) {
    // First compare y coordinate, then compare x coordinate
    if (box1[0].y == box2[0].y) {
        return box1[0].x < box2[0].x;
    }
    return box1[0].y < box2[0].y;
}

std::vector<std::vector<cv::Point>> sorted_boxes(const std::vector<std::vector<cv::Point>>& dt_boxes) {
    int num_boxes = dt_boxes.size();

    // Step 1: Sort by top-left corner (y then x)
    std::vector<std::vector<cv::Point>> sorted_boxes = dt_boxes;
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), sort_by_top_left);

    // Step 2: Apply second sorting to ensure order from top to bottom, left to right
    for (int i = 0; i < num_boxes - 1; ++i) {
        for (int j = i; j >= 0; --j) {
            // Compare vertical distances between boxes to check proximity
            if (std::abs(sorted_boxes[j + 1][0].y - sorted_boxes[j][0].y) < 10) {
                if (sorted_boxes[j + 1][0].x < sorted_boxes[j][0].x) {
                    std::swap(sorted_boxes[j], sorted_boxes[j + 1]);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    return sorted_boxes;
}

// 转换函数
std::vector<std::vector<int>> convertBoxes(const std::vector<std::vector<cv::Point>>& boxes) {
    std::vector<std::vector<int>> result;
    result.reserve(boxes.size());

    for (const auto& box : boxes) {
        if (box.size() < 2) continue; // 确保至少有两个点

        int x_min = (int)box[0].x, y_min = (int)box[0].y;
        int x_max = (int)box[0].x, y_max = (int)box[0].y;

        // 计算最小和最大 x、y 坐标
        for (const auto& point : box) {
            x_min = std::min(x_min, (int)point.x);
            y_min = std::min(y_min, (int)point.y);
            x_max = std::max(x_max, (int)point.x);
            y_max = std::max(y_max, (int)point.y);
        }

        result.push_back({x_min, y_min, x_max, y_max});
    }

    return result;
}