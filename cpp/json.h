#pragma once

#include <map>
#include <string>

bool read_json_str_dict(const std::string &input,
                        std::map<std::string, std::string> &output);
void write_json_str_dict(const std::map<std::string, std::string> &input,
                         std::string &output);
