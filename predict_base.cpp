//
// Created by admin on 2025/2/25.
//

#include "predict_base.h"

// 获取输出节点名称
std::vector<std::string> PredictBase::getOutputName(Ort::Session& session) {
    std::vector<std::string> output_names;
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; ++i) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names.emplace_back(output_name.get()); // 将名称存储为 std::string
    }
    return output_names;
}

// 获取输入节点名称
std::vector<std::string> PredictBase::getInputName(Ort::Session& session) {
    std::vector<std::string> input_names;
    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; ++i) {
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_names.emplace_back(input_name.get()); // 将名称存储为 std::string
    }
    return input_names;
}
