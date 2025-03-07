//
// Created by admin on 2025/3/5.
//

#ifndef PADDLEOCR_OCR_H
#define PADDLEOCR_OCR_H
#include "utils.h"
#include "predict_det.h"
#include "predict_rec.h"

struct OcrResults {
    std::vector<const char*> resTexts;
    std::vector<std::vector<int>> resBoxes = {};
};
extern "C" __declspec(dllexport) void* initDetModel(const char* detModelPath, bool useGpu);
extern "C" __declspec(dllexport) void* initRecModel(const char* recModelPath, const char* dictPath, bool useGpu);
extern "C" __declspec(dllexport) OcrResults* Infer(void* textDetectorPtr, void* textRecognizerPtr, unsigned char* imageData, int width, int height);
extern "C" __declspec(dllexport) void freeOcrResults(OcrResults* results);
extern "C" __declspec(dllexport) int get_box_count(OcrResults* result);
extern "C" __declspec(dllexport) int get_box_size(OcrResults* result, int i);
extern "C" __declspec(dllexport) int get_text_count(OcrResults* result);
extern "C" __declspec(dllexport) const char* get_text(OcrResults* result, int i);
extern "C" __declspec(dllexport) int* get_box(OcrResults* result, int i);
#endif //PADDLEOCR_OCR_H
