#!/bin/bash

set -eu

isort .
black -l 100 .
flake8 . --ignore E501
