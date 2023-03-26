#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "utils.h"

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    {4096, 1},
    {5120, 1},
    {6656, 1},
    {8192, 1},
};

// default hparams (LLaMA 7B)
struct llama_hparams {
  int32_t n_vocab = 32000;
  int32_t n_ctx = 512; // this is provided as user input?
  int32_t n_embd = 4096;
  int32_t n_mult = 256;
  int32_t n_head = 32;
  int32_t n_layer = 32;
  int32_t n_rot = 64;
  int32_t f16 = 1;
};

struct llama_layer {
  // normalization
  struct ggml_tensor *attention_norm;

  // attention
  struct ggml_tensor *wq;
  struct ggml_tensor *wk;
  struct ggml_tensor *wv;
  struct ggml_tensor *wo;

  // normalization
  struct ggml_tensor *ffn_norm;

  // ff
  struct ggml_tensor *w1;
  struct ggml_tensor *w2;
  struct ggml_tensor *w3;
};

struct llama_model {
  llama_hparams hparams;

  struct ggml_tensor *tok_embeddings;

  struct ggml_tensor *norm;
  struct ggml_tensor *output;

  std::vector<llama_layer> layers;

  // key + value memory
  struct ggml_tensor *memory_k;
  struct ggml_tensor *memory_v;

  //
  struct ggml_context *ctx;
  std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool llama_model_load(const std::string &fname, llama_model &model,
                      gpt_vocab &vocab, int n_ctx, bool verbose);

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool llama_eval(const llama_model &model, const int n_threads, const int n_past,
                const std::vector<gpt_vocab::id> &embd_inp,
                std::vector<float> &embd_w, size_t &mem_per_token);

bool llama_eval_t(const llama_model &model, const int n_threads,
                  const int n_past, const std::vector<gpt_vocab::id> &embd_inp,
                  std::vector<float> &embd_w, size_t &mem_per_token,
                  int64_t &duration_us);

std::map<std::string, std::string> llama_system_info(void);