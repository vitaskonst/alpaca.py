#! /bin/bash

set -eu

clang-format -i cpp/*.cpp cpp/*.h cpp/*.c
