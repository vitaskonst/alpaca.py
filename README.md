# Alpaca.py

A python client based on [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp).

The most important change W.R.T the original code is that context is not maintained between calls. That is there is no state, so by default it doesn't behave like a chat bot. That said, it's easy to add that simply by keeping track of all user and system utterances.

## Build

Build cpp binary:

```shell
cd cpp
mkdir -p build
make
cd ..
```

Set up python environment:

```shell
conda create -n alpaca python=3.8
pip install -r requirements.txt 
pip install streamlit==1.20.0  # For the streamlit demo below.
conda activate alpaca
```

## Try it out

### Command line

```shell
python demo_cli.py --alpaca-cli-path cpp/build/alpaca --model-path $MODEL_DIR/ggml-alpaca-7b-q4.bin 
```

### Web Demo (Streamlit):

(This requires installing streamlit (see above).)

```shell
export ALPACA_CLI_PATH="$PWD/cpp/build/alpaca"
export ALPACA_MODEL_PATH="$MODEL_DIR/ggml-alpaca-7b-q4.bin"
streamlit run demo_st.py 
```

### JSON REST Api (FastApi)

Navigate to http://127.0.0.1:8080/docs for the docs after starting it.

```shell
uvicorn alpaca_api:app --port 8080
```

### Python Module:

```python
from alpaca import Alpaca, InferenceRequest

alpaca = Alpaca(alpaca_cli_path, model_path)
try:
    output = alpaca.run_simple(InferenceRequest(input_text="Are alpacas afraid of snakes?"))["output"]
finally:
    alpaca.stop()
```

## Docker

Launches the JSON API via docker.
Navigate to http://127.0.0.1:8080/docs for the docs after starting it.

```shell
docker build . -f docker/Dockerfile -t alpaca_api
docker run --name alpaca_api --mount type=bind,source="$MODEL_DIR",target=/models -p 8080:8080 -d alpaca_api
```

## Credit

This is a fork of [alpaca.cpp](https://github.com/antimatter15/alpaca.cpp), which itself gives the following credit:
> This combines [Facebook's LLaMA](https://github.com/facebookresearch/llama), [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [alpaca-lora](https://github.com/tloen/alpaca-lora) and [corresponding weights](https://huggingface.co/tloen/alpaca-lora-7b/tree/main) by Eric Wang (which uses [Jason Phang's implementation of LLaMA](https://github.com/huggingface/transformers/pull/21955) on top of Hugging Face Transformers), and [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov. The chat implementation is based on Matvey Soloviev's [Interactive Mode](https://github.com/ggerganov/llama.cpp/pull/61) for llama.cpp. Inspired by [Simon Willison's](https://til.simonwillison.net/llms/llama-7b-m2) getting started guide for LLaMA. [Andy Matuschak](https://twitter.com/andy_matuschak/status/1636769182066053120)'s thread on adapting this to 13B, using fine tuning weights by [Sam Witteveen](https://huggingface.co/samwit/alpaca13B-lora). 
