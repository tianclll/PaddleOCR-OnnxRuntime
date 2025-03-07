#include <filesystem>
#include "OCR.h"

int main() {
    // 读取图像
    auto img = cv::imread("C:\\Users\\admin\\Desktop\\PaddleOCR\\test_images\\00006737.jpg");
    const char* detModelPath = "C:\\Users\\admin\\Desktop\\PaddleOCR\\weights\\det.onnx";
    const char* recModelPath = "C:\\Users\\admin\\Desktop\\PaddleOCR\\weights\\rec.onnx";
    const char* dictPath = "C:\\Users\\admin\\Desktop\\PaddleOCR\\ch.txt";
    bool useGpu = true;
    // 获取图像数据
    unsigned char* imageData = img.data;
    int width = img.cols;
    int height = img.rows;

    // 初始化模型
    void* textDetectorPtr = initDetModel(detModelPath, useGpu);
    void* textRecognizerPtr = initRecModel(recModelPath, dictPath, useGpu);

    // 模型推理
    //推理文件夹中的图片
    //遍历文件夹中的所有图片
    for (const auto& entry : std::filesystem::directory_iterator("C:\\Users\\admin\\Desktop\\PaddleOCR\\test_images")) {
        auto img = cv::imread(entry.path().string());
        unsigned char* imageData = img.data;
        int width = img.cols;
        int height = img.rows;
        //计算推理时间
        auto start = std::chrono::high_resolution_clock::now();
        auto* result = Infer(textDetectorPtr, textRecognizerPtr, imageData, width, height);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Inference time: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        // 释放资源
        // 假设 resBoxes 是一个动态分配的数组，每个元素是一个 4 元素的数组
        for (int i = 0; i < result->resBoxes.size(); i++) {
            // 访问每个框的数据，确保是四个整数
//            std::cout << "Box " << i << ": [";
//            for (int j = 0; j < 4; j++) {
//                std::cout << result->resBoxes[i][j];
//                if (j < 3) {
//                    std::cout << ", ";
//                }
//            }
//        std::cout << "]" << std::endl;
        std::cout << "Text: " << result->resTexts[i] << std::endl;
    }
        freeOcrResults(result);
    }
    return 0;
}
