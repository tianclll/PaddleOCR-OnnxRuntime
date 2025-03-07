//
// Created by admin on 2025/3/4.
//

#include "rec_postprocess.h"

BaseRecLabelDecode::BaseRecLabelDecode(const std::string& character_dict_path, bool use_space_char)
        : beg_str("sos"), end_str("eos"), reverse(false) {

    if (character_dict_path.empty()) {
        character_str = "0123456789abcdefghijklmnopqrstuvwxyz";
        for (char c : character_str) {
            dict_character.emplace_back(1, c); // 将单个字符转换为 std::string 并存入 vector
        }
    } else {
        std::ifstream file(character_dict_path);
        std::string line;
        while (std::getline(file, line)) {
            dict_character.push_back(line);
        }
        if (use_space_char) {
            dict_character.push_back(" ");
        }
        if (character_dict_path.find("arabic") != std::string::npos) {
            reverse = true;
        }
    }

    dict_character = add_special_char(dict_character);
    for (size_t i = 0; i < dict_character.size(); ++i) {
        dict[dict_character[i]] = i;
    }
}

std::vector<std::string> BaseRecLabelDecode::add_special_char(const std::vector<std::string>& dict_character) {
    return dict_character;
}

std::vector<std::pair<std::string, float>> BaseRecLabelDecode::decode(const std::vector<std::vector<int>>& text_index,
                                                  const std::vector<std::vector<float>>& text_prob,
                                                  bool is_remove_duplicate) {
    std::vector<std::pair<std::string, float>> result_list;
    std::vector<int> ignored_tokens = get_ignored_tokens();

    for (size_t batch_idx = 0; batch_idx < text_index.size(); ++batch_idx) {
        std::vector<int> selection(text_index[batch_idx].size(), 1);
        if (is_remove_duplicate) {
            for (size_t i = 1; i < text_index[batch_idx].size(); ++i) {
                if (text_index[batch_idx][i] == text_index[batch_idx][i-1]) {
                    selection[i] = 0;
                }
            }
        }

        for (int ignored_token : ignored_tokens) {
            for (size_t i = 0; i < text_index[batch_idx].size(); ++i) {
                if (text_index[batch_idx][i] == ignored_token) {
                    selection[i] = 0;
                }
            }
        }

        std::string text;
        std::vector<float> conf_list;
        for (size_t i = 0; i < text_index[batch_idx].size(); ++i) {
            if (selection[i]) {
                text += this->dict_character[text_index[batch_idx][i] - 1];
                 conf_list.push_back((text_prob)[batch_idx][i]);
            }
        }

        if (conf_list.empty()) {
            conf_list.push_back(0.0f);
        }

        float avg_conf = std::accumulate(conf_list.begin(), conf_list.end(), 0.0f) / conf_list.size();

        if (reverse) {
            text = pred_reverse(text);
        }

        result_list.emplace_back(text, avg_conf);
    }
    return result_list;
}

std::string BaseRecLabelDecode::pred_reverse(const std::string& pred) {
    std::string reversed_pred;
    std::string c_current;
    for (char c : pred) {
        if (!isalnum(c) && c != ' ' && c != ':' && c != '*' && c != '.' && c != '/' && c != '%' && c != '+' && c != '-') {
            if (!c_current.empty()) {
                reversed_pred += c_current;
            }
            reversed_pred += c;
            c_current.clear();
        } else {
            c_current += c;
        }
    }
    if (!c_current.empty()) {
        reversed_pred += c_current;
    }
    std::reverse(reversed_pred.begin(), reversed_pred.end());
    return reversed_pred;
}

std::vector<int> BaseRecLabelDecode::get_ignored_tokens() {
    return {0};
}


std::vector<std::string> CTCLabelDecode::add_special_char(const std::vector<std::string>& dict_character) {
    std::vector<std::string> new_dict = {"blank"};
    new_dict.insert(new_dict.end(), dict_character.begin(), dict_character.end());
    return new_dict;
}

std::vector<std::pair<std::string, float>> CTCLabelDecode::operator()(std::vector<Ort::Value>& output_tensors) {
    // 获取输出张量形状
    auto& output_tensor = output_tensors.at(0);
    auto shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3) {
        throw std::runtime_error("Output tensor must have shape [batch, seq_len, num_classes]");
    }
    const int batch_size = shape[0];
    const int seq_len = shape[1];
    const int num_classes = shape[2];
    const float* pred_data = output_tensor.GetTensorData<float>();

    // 并行处理每个批次
    std::vector<std::vector<int>> preds_idx(batch_size, std::vector<int>(seq_len, 0));
    std::vector<std::vector<float>> preds_prob(batch_size, std::vector<float>(seq_len, 0.0f));

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            int max_index = 0;
            float max_prob = pred_data[i * seq_len * num_classes + j * num_classes];

            for (int k = 1; k < num_classes; ++k) {
                const float prob = pred_data[i * seq_len * num_classes + j * num_classes + k];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_index = k;

                }
            }

            preds_idx[i][j] = max_index;
            preds_prob[i][j] = max_prob;
        }
    }

    return decode(preds_idx, preds_prob, true);
}

