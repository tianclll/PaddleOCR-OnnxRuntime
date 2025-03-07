//
// Created by admin on 2025/3/4.
//

#ifndef PADDLEOCR_REC_POSTPROCESS_H
#define PADDLEOCR_REC_POSTPROCESS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <onnxruntime_cxx_api.h>
class BaseRecLabelDecode {
public:
    explicit BaseRecLabelDecode(const std::string& character_dict_path, bool use_space_char = false);
    virtual std::vector<std::string> add_special_char(const std::vector<std::string>& dict_character);
    std::vector<std::pair<std::string, float>> decode(const std::vector<std::vector<int>>& text_index,
                                                      const std::vector<std::vector<float>>& text_prob,
                                                      bool is_remove_duplicate = false);
protected:
    std::string beg_str, end_str;
    bool reverse;
    std::string character_str;
    std::vector<std::string> dict_character {};
    std::unordered_map<std::string, int> dict;
    std::string pred_reverse(const std::string& pred);

    virtual std::vector<int> get_ignored_tokens();
};
class CTCLabelDecode : public BaseRecLabelDecode {
public:
    explicit CTCLabelDecode(const std::string& character_dict_path = "", bool use_space_char = true)
            : BaseRecLabelDecode(character_dict_path, use_space_char) {};
    std::vector<std::string> add_special_char(const std::vector<std::string>& dict_character) override;
    std::vector<std::pair<std::string, float>> operator()(std::vector<Ort::Value>& output_tensors);
};
#endif //PADDLEOCR_REC_POSTPROCESS_H
