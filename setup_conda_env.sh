#!/usr/bin/env bash
set -e

ENV_NAME="chb_env"
PYTHON_VERSION="3.10"

# Create conda environment with specified Python version
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# Install Python dependencies from requirements.txt
conda run -n "$ENV_NAME" pip install --upgrade pip
conda run -n "$ENV_NAME" pip install -r requirements.txt

echo "Environment '$ENV_NAME' created. Activate with: conda activate $ENV_NAME"
