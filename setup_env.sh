#!/bin/bash

# Exit immediately if a command fails
set -e

# apt update
# apt install -y python3.9 python3.9-venv python3.9-dev

# Choose your Python version
PYTHON_VERSION=3.9
ENV_NAME=~/controlnet-env

echo "Creating virtual environment with Python $PYTHON_VERSION..."

# Check if the requested python version exists
if ! command -v python$PYTHON_VERSION &> /dev/null
then
    echo "Python $PYTHON_VERSION is not installed. Please install it and try again."
    exit 1
fi

# Create virtual environment
python$PYTHON_VERSION -m venv $ENV_NAME

# Activate environment
source $ENV_NAME/bin/activate

echo "Environment '$ENV_NAME' created and activated."

# Upgrade pip and install dependencies
pip install --upgrade pip

echo "Installing required Python packages..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers[training] transformers accelerate datasets
pip install pillow tqdm

echo "✅ All dependencies installed successfully."
echo "💡 To activate the environment again later, run: source $ENV_NAME/bin/activate"
