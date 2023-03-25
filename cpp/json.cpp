#include "json.h"

bool read_next_token(const std::string &input, size_t &i, std::string &token) {
  token.clear();
  while (i < input.size()) {

    if (input[i] == ' ') {
      ++i;
      continue;
    }

    if (input[i] != '"') {
      token.append(1, input[i]);
      i++;
      return true;
    }

    // Read string literal.
    i++;
    while (i < input.size()) {
      if (input[i] == '"') {
        i++;
        return true;
      }

      if (input[i] == '\\') {
        i++;
        if (i >= input.size()) {
          // Invalid string literal.
          return false;
        }
        switch (input[i]) {
        case 't':
          token.append(1, '\t');
          break;
        case 'n':
          token.append(1, '\n');
          break;
        case '\\':
          token.append(1, '\\');
          break;
        case '"':
          token.append(1, '"');
          break;
        default:
          // Invalid string literal.
          return false;
        }
      } else {
        token.append(1, input[i]);
      }
      i++;
    }
    // Invalid string literal (no closing double quote).
    return false;
  }
  // Reached end of string.
  return false;
}

bool read_json_str_dict(const std::string &input,
                        std::map<std::string, std::string> &output) {
  size_t i = 0;
  std::string token;

  if (!read_next_token(input, i, token)) {
    return false;
  }

  if (token != "{") {
    return false;
  }

  while (true) {

    std::string key;
    if (!read_next_token(input, i, key)) {
      return false;
    }
    if (!read_next_token(input, i, token)) {
      return false;
    }
    if (token != ":") {
      return false;
    }
    std::string value;
    if (!read_next_token(input, i, value)) {
      return false;
    }
    output[key] = value;
    if (!read_next_token(input, i, token)) {
      return false;
    }
    if (token == ",") {
      continue;
    }
    if (token == "}") {
      break;
    }
    return false;
  }

  return true;
}

void write_string(const std::string &value, std::string &output) {
  output.append(1, '"');
  for (size_t i = 0; i < value.size(); ++i) {
    switch (value[i]) {
    case '"':
      output.append(1, '\\');
      output.append(1, '"');
      break;
    case '\t':
      output.append(1, '\\');
      output.append(1, 't');
      break;
    case '\n':
      output.append(1, '\\');
      output.append(1, 'n');
      break;
    case '\\':
      output.append(1, '\\');
      output.append(1, '\\');
      break;
    default:
      output.append(1, value[i]);
    }
  }
  output.append(1, '"');
}

void write_json_str_dict(const std::map<std::string, std::string> &input,
                         std::string &output) {
  output.append(1, '{');
  bool needs_comma = false;
  for (auto &p : input) {
    if (needs_comma) {
      output.append(1, ',');
    }
    const std::string &key = p.first;
    const std::string &value = p.second;
    write_string(key, output);
    output.append(1, ':');
    write_string(value, output);
    needs_comma = true;
  }
  output.append(1, '}');
}
