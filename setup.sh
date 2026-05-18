#!/bin/bash

set -eou pipefail

# Create virtual environment and install dependencies
uv venv
uv sync

# Download DreamBooth dataset
# uv run python scripts/download_datasets.py --dataset all  # Download all datasets
uv run python scripts/download_datasets.py --dataset dreambooth

# Symlink check
ln -sv /workspace/kunkim/experiments/textboost ${PWD}/outputs
ln -sv /workspace/kunkim/models/hf ${PWD}/models
