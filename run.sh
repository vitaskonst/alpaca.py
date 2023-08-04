#!/bin/bash

export MODEL_DIR="$PWD/../alpaca.model"
export KEYS_PATH="$PWD/keys.txt"
docker build . -f docker/Dockerfile -t alpaca_api
docker run \
--name alpaca_api \
--mount type=bind,source="$MODEL_DIR",target=/models \
--mount type=bind,source="$KEYS_PATH",target=/code/keys.txt \
--env KEYS_PATH=/code/keys.txt \
-p 80:8080 \
-d alpaca_api