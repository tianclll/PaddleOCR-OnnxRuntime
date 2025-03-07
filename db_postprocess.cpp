#include "db_postprocess.h"

std::vector<std::vector<cv::Point>> DBPostProcess::boxes_from_bitmap(const cv::Mat& pred, const cv::Mat& bitmap, int dest_width, int dest_height) {
    std::vector<std::vector<cv::Point>> boxes;
    std::vector<float> scores;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
//        cv::drawContours(pred, contours, -1, cv::Scalar(0, 255, 0), 2);

    int num_contours = std::min(contours.size(), static_cast<size_t>(max_candidates));

    for (int i = 0; i < num_contours; ++i) {
        std::vector<cv::Point> contour = contours[i];
        auto [points, sside] = get_mini_boxes(contour);
        if (sside < this->min_size) continue;

        float score = (this->score_mode == "fast") ? box_score_fast(pred, points) : box_score_slow(pred, contour);
        if (this->box_thresh > score) continue;

        std::vector<cv::Point> box = unclip(points, this->unclip_ratio);
        auto [box_points, box_sside] = get_mini_boxes(box);
        if (box_sside < this->min_size + 2) continue;
        for (auto& point : box_points) {
            point.x = std::clamp(static_cast<int>(std::round(point.x / static_cast<double>(bitmap.cols) * dest_width)), 0, dest_width);
            point.y = std::clamp(static_cast<int>(std::round(point.y / static_cast<double>(bitmap.rows) * dest_height)), 0, dest_height);
        }
        boxes.push_back(box_points);
        scores.push_back(score);
    }
    return boxes;
}

std::vector<cv::Point> DBPostProcess::unclip(const std::vector<cv::Point>& box, float unclip_ratio) {
    ClipperLib::Path path;
    for (const auto& point : box) {
        path << ClipperLib::IntPoint(point.x, point.y);
    }

    // 计算面积

    double areaValue = std::abs(ClipperLib::Area(path));

    double area = (areaValue < 0) ? -areaValue : areaValue;
    if (area < 1e-6) {
        return box; // Avoid division by zero or invalid operations
    }

    double perimeter = 0.0;
    for (size_t i = 0; i < path.size(); ++i) {
        size_t j = (i + 1) % path.size();
        double dx = path[j].X - path[i].X;
        double dy = path[j].Y - path[i].Y;
        perimeter += std::sqrt(dx * dx + dy * dy);
    }

    double distance = area * unclip_ratio / perimeter;

    ClipperLib::ClipperOffset offset;
    offset.AddPath(path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths expanded;
    offset.Execute(expanded, distance);

    std::vector<cv::Point> result;
    if (!expanded.empty()) {
        for (const auto& p : expanded[0]) {
            result.emplace_back(p.X, p.Y);
        }
    }
    return result;
}

std::pair<std::vector<cv::Point>, float> DBPostProcess::get_mini_boxes(const std::vector<cv::Point>& contour) {
    cv::RotatedRect bounding_box = cv::minAreaRect(contour);
//         使用 cv::Point2f 存储顶点
    std::vector<cv::Point2f> points(4);
    bounding_box.points(points.data());
    std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
    if (points[1].y > points[0].y) {
        index_1 = 0;
        index_4 = 1;
    } else {
        index_1 = 1;
        index_4 = 0;
    }
    if (points[3].y > points[2].y) {
        index_2 = 2;
        index_3 = 3;
    } else {
        index_2 = 3;
        index_3 = 2;
    }

    std::vector<cv::Point> box = {points[index_1], points[index_2], points[index_3], points[index_4]};
    return {box, std::min(bounding_box.size.width, bounding_box.size.height)};
}

float DBPostProcess::box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point>& box) {
    int h = bitmap.rows;
    int w = bitmap.cols;
    int xmin_ = std::min(box[0].x, box[3].x);
    int xmax_ = std::max(box[1].x, box[2].x);
    int ymin_ = std::min(box[0].y, box[1].y);
    int ymax_ = std::max(box[2].y, box[3].y);
    int xmin = std::clamp(static_cast<int>(std::floor(xmin_)), 0, w - 1);
    int xmax = std::clamp(static_cast<int>(std::ceil(xmax_)), 0, w - 1);
    int ymin = std::clamp(static_cast<int>(std::floor(ymin_)), 0, h - 1);
    int ymax = std::clamp(static_cast<int>(std::ceil(ymax_)), 0, h - 1);

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<cv::Point> adjusted_box = box;
    for (auto& point : adjusted_box) {
        point.x -= xmin;
        point.y -= ymin;
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{adjusted_box}, 1);
    return cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask)[0];
}

float DBPostProcess::box_score_slow(const cv::Mat& bitmap, const std::vector<cv::Point>& contour) {
    int h = bitmap.rows;
    int w = bitmap.cols;

    int xmin = std::clamp(static_cast<int>(std::min_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; })->x), 0, w - 1);
    int xmax = std::clamp(static_cast<int>(std::max_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; })->x), 0, w - 1);
    int ymin = std::clamp(static_cast<int>(std::min_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; })->y), 0, h - 1);
    int ymax = std::clamp(static_cast<int>(std::max_element(contour.begin(), contour.end(), [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; })->y), 0, h - 1);

    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
    std::vector<cv::Point> adjusted_contour = contour;
    for (auto& point : adjusted_contour) {
        point.x -= xmin;
        point.y -= ymin;
    }
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{adjusted_contour}, 1);
    return cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask)[0];
};

std::vector<std::vector<cv::Point>> DBPostProcess::operator()(const cv::Mat& preds, const std::tuple<int, int, float, float>& shape_list) {
    std::vector<std::vector<cv::Point>> boxes_batch;

    cv::Mat pred = preds.clone();
    cv::Mat segmentation = pred > thresh;

    int src_h = std::get<0>(shape_list);
    int src_w = std::get<1>(shape_list);
    float ratio_h = std::get<2>(shape_list);
    float ratio_w = std::get<3>(shape_list);

    cv::Mat mask;
    mask = segmentation;
    std::vector<std::vector<cv::Point>> boxes;
    boxes = boxes_from_bitmap(pred, mask, src_w, src_h);

    return boxes;
};