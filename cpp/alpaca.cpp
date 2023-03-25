#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "json.h"
#include "llama.h"

enum ReadState {
  READ_STATE_SUCCESS = 0,
  READ_STATE_STOP = 1,
  READ_STATE_ERROR = 2
};

ReadState read_input(const gpt_vocab &vocab, std::vector<gpt_vocab::id> &input,
                     gpt_params &params) {
  char buffer[4096] = {0};
  int n_read;

  if (scanf("%4095[^\n]%n%*c", buffer, &n_read) <= 0) {
    // presumable empty line, consume the newline
    if (scanf("%*c") <= 0) { /*ignore*/
    }
    n_read = 0;
  }

  if (n_read == 0) {
    return READ_STATE_ERROR;
  }

  for (auto quit_cmd : {"exit();", "quit();"}) {
    if (strcmp(buffer, quit_cmd) == 0) {
      if (params.verbose)
        fprintf(stderr, "returning stop\n");
      return READ_STATE_STOP;
    }
  }

  std::map<std::string, std::string> args;
  if (!read_json_str_dict(buffer, args)) {
    return READ_STATE_ERROR;
  }

  std::string input_text = "";

  for (auto &arg : args) {
    auto &key = arg.first;
    auto &value = arg.second;
    if (key == "input_text") {
      input_text = value;
    } else if (key == "seed") {
      params.seed = std::stoi(value);
    } else if (key == "n_predict") {
      params.n_predict = std::stoi(value);
    } else if (key == "top_k") {
      params.top_k = std::stoi(value);
    } else if (key == "top_p") {
      params.top_p = std::stof(value);
    } else if (key == "temp") {
      params.temp = std::stof(value);
    } else if (key == "repeat_last_n") {
      params.repeat_last_n = std::stoi(value);
    } else if (key == "repeat_penalty") {
      params.repeat_penalty = std::stof(value);
    } else {
      if (params.verbose)
        fprintf(stderr, "Unknown argument: %s\n", key.c_str());
      return READ_STATE_ERROR;
    }
  }

  std::vector<gpt_vocab::id> tokens =
      ::llama_tokenize(vocab, input_text, false);
  input.insert(input.end(), tokens.begin(), tokens.end());
  return READ_STATE_SUCCESS;
}

void write_error(const std::string &message) {
  fprintf(stdout, "{ \"error\": \"%s\" }\n", message.c_str());
  fflush(stdout);
}

int main(int argc, char **argv) {
  ggml_time_init();
  const int64_t t_main_start_us = ggml_time_us();

  gpt_params global_params;

  if (gpt_params_parse(argc, argv, global_params) == false) {
    return 1;
  }

  gpt_vocab vocab;
  llama_model model;

  // load the model
  {
    if (!llama_model_load(global_params.model, model, vocab,
                          global_params.n_ctx, global_params.verbose)) {
      fprintf(stderr, "ERROR: failed to load model from '%s'\n",
              global_params.model.c_str());
      return 1;
    }
  }

  // print system information
  if (global_params.verbose) {
    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            global_params.n_threads, std::thread::hardware_concurrency(),
            llama_print_system_info());
  }

  // Determine the required inference memory per token:
  size_t mem_per_token = 0;
  {
    std::vector<float> logits;
    llama_eval(model, global_params.n_threads, 0, {0, 1, 2, 3}, logits,
               mem_per_token);
  }

  while (true) {
    int64_t t_predict_us = 0;

    gpt_params params(global_params);
    if (params.verbose)
      fprintf(stderr, "Reading input ...\n");
    std::vector<gpt_vocab::id> embd_inp;
    ReadState state = read_input(vocab, embd_inp, params);
    if (state == READ_STATE_STOP) {
      if (params.verbose)
        fprintf(stderr, "Stopping program\n");
      break;
    }
    if (state == READ_STATE_ERROR) {
      if (params.verbose)
        write_error("Invalid arguments");
      continue;
    }

    if (params.seed < 0) {
      params.seed = time(NULL);
    }
    std::mt19937 rng(params.seed);

    size_t n_tokens_truncated = 0;
    while (embd_inp.size() + params.n_predict > model.hparams.n_ctx) {
      embd_inp.erase(embd_inp.begin());
      ++n_tokens_truncated;
    }

    if (embd_inp.size() > model.hparams.n_ctx) {
      write_error("Input exceeds maximal context length");
      continue;
    }

    if (params.verbose)
      fprintf(stderr, "Embedding input ...\n");
    // Embedding input.
    size_t n_past = 0;
    std::vector<float> logits;
    {
      int64_t duration_us;
      if (!llama_eval_t(model, params.n_threads, /*n_past=*/0, embd_inp, logits,
                        mem_per_token, duration_us)) {
        write_error("Inference error");
        return 1;
      }
      t_predict_us += duration_us;
      n_past += embd_inp.size();
    }

    if (params.verbose)
      fprintf(stderr, "Predicting outputs ...\n");
    // Predicting outputs.
    std::string output_text;
    std::vector<gpt_vocab::id> last_n_tokens(params.repeat_last_n);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    bool inference_error = false;
    size_t n_outputs = 0;
    while (n_past < model.hparams.n_ctx) {
      gpt_vocab::id id = llama_sample_top_p_top_k(
          vocab, logits.data(), last_n_tokens, params.repeat_penalty,
          params.top_k, params.top_p, params.temp, rng);
      output_text.append(vocab.id_to_token[id]);
      n_outputs++;
      if (id == 2 || n_outputs >= params.n_predict) {
        break;
      }
      {
        int64_t duration_us;
        if (!llama_eval_t(model, params.n_threads,
                          /*n_past=*/n_past,
                          /*embd_inp=*/{id}, logits, mem_per_token,
                          duration_us)) {
          write_error("Inference error");
          inference_error = true;
          break;
        }
        t_predict_us += duration_us;
        n_past += 1;
      }
      last_n_tokens.erase(last_n_tokens.begin());
      last_n_tokens.push_back(id);
    }
    if (inference_error) {
      continue;
    }

    if (params.verbose)
      fprintf(stderr, "Converting outputs ...\n");
    std::map<std::string, std::string> output_map;
    output_map["output"] = output_text;
    output_map["reached_max_content_size"] =
        ((n_past == model.hparams.n_ctx) ? "true" : "false");
    {
      char buffer[256];
      sprintf(buffer, "%8zu", mem_per_token);
      output_map["memory_per_token_bytes"] = buffer;
      sprintf(buffer, "%lu", t_predict_us);
      output_map["total_predict_time_us"] = buffer;
      sprintf(buffer, "%lu", n_past);
      output_map["total_token_length"] = buffer;
      sprintf(buffer, "%lu", n_tokens_truncated);
      output_map["n_tokens_truncated"] = buffer;
    }

    if (params.verbose)
      fprintf(stderr, "Writing outputs ...\n");
    std::string output_json;
    write_json_str_dict(output_map, output_json);
    fprintf(stdout, "%s", output_json.c_str());
    fprintf(stdout, "\n");
    fflush(stdout);
  }

  ggml_free(model.ctx);
  return 0;
}
