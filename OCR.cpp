//
// Created by admin on 2025/3/5.
//
#include "OCR.h"

void* initDetModel(const char* detModelPath, bool useGpu){
    return new TextDetector(detModelPath, useGpu);
}
void* initRecModel(const char* recModelPath, const char* dictPath, bool useGpu){
    return new TextRecognizer(recModelPath, dictPath, useGpu);
}

OcrResults* Infer(void* textDetectorPtr, void* textRecognizerPtr, unsigned char* imageData, int width, int height){
    cv::Mat img(height, width, CV_8UC3, imageData);
    // 先转换指针类型
    auto* textDetector = static_cast<TextDetector*>(textDetectorPtr);
    auto* textRecognizer = static_cast<TextRecognizer*>(textRecognizerPtr);
    std::vector<std::vector<cv::Point>> boxes = textDetector->operator()(img);
    boxes = sorted_boxes(boxes);
    std::vector<cv::Mat> image_list;
    for (auto &box : boxes) {
        cv::Mat crop_image = getMinAreaRectCrop(img, box);
        image_list.push_back(crop_image);
    }
    std::vector<std::pair<std::string, float>> res = textRecognizer->operator()(image_list);
    std::vector<const char*> resTexts ;
    std::vector<std::vector<cv::Point>> resBoxes;
    for (int i = 0; i < res.size(); ++i) {
        float score = res[i].second;
        if (score >= 0.5) {
            char* text = new char[res[i].first.size() + 1];  // 动态分配
            strcpy(text, res[i].first.c_str());  // 复制字符串
            resTexts.push_back(text);
            resBoxes.push_back(boxes[i]);
//            std::cout << "text: " <<Text << ", score: " << res[i].second << std::endl;
        }
    }
    std::vector<std::vector<int>> resBoxesInt = convertBoxes(resBoxes);
    auto* resOCR = new OcrResults();
    resOCR->resTexts = resTexts;
    resOCR->resBoxes = resBoxesInt;
    return resOCR;
}
int get_box_count(OcrResults* result) {
    return result->resBoxes.size();
}
int get_box_size(OcrResults* result, int i) {
    return result->resBoxes[i].size();
}
int get_text_count(OcrResults* result) {
    return result->resTexts.size();
}
const char* get_text(OcrResults* result, int i) {
    return result->resTexts[i];
}
int* get_box(OcrResults* result, int i) {
    return result->resBoxes[i].data();  // 返回指向 vector<int> 的指针
}
// 释放 OCR 结果
void freeOcrResults(OcrResults* results) {
    for (const char* text : results->resTexts) {
        delete[] text;
    }
    if (results) {
        delete results;
    }
}
