FROM python:3.8-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends build-essential

ADD requirements.txt requirements.txt

RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ADD cpp cpp

WORKDIR /code/cpp

RUN mkdir build && make

WORKDIR /code

ADD alpaca.py alpaca.py

ADD alpaca_api.py alpaca_api.py

ENV ALPACA_CLI_PATH="/code/cpp/build/alpaca"

ENV ALPACA_MODEL_PATH="/models/ggml-alpaca-7b-q4.bin"

EXPOSE 8080

ENTRYPOINT uvicorn alpaca_api:app --host 0.0.0.0 --port 8080