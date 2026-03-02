#!/bin/bash

# Script to build machine unlearning project environment
# Usage: bash build_env.sh

set -e  # Exit immediately on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Machine Unlearning Project Environment Setup Script${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda command not found${NC}"
    echo -e "${YELLOW}Please install Anaconda or Miniconda first${NC}"
    exit 1
fi

# Set environment name
ENV_NAME="unlearning-test"
PYTHON_VERSION="3.12"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Warning: Environment '${ENV_NAME}' already exists${NC}"
    read -p "Delete and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        conda env remove -n ${ENV_NAME} -y
    else
        echo -e "${RED}Operation cancelled${NC}"
        exit 1
    fi
fi

# Create new conda environment
echo -e "${GREEN}Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})${NC}"
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install project dependencies
echo -e "${GREEN}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Verify installation
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Verifying installation...${NC}"
echo -e "${GREEN}======================================${NC}"

python -c "
import sys
print(f'Python version: {sys.version}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')

import numpy as np
print(f'NumPy version: {np.__version__}')

import pandas as pd
print(f'Pandas version: {pd.__version__}')

import sklearn
print(f'Scikit-learn version: {sklearn.__version__}')

import fairlearn
print(f'Fairlearn version: {fairlearn.__version__}')

print('\\nAll core dependencies installed successfully!')
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Environment setup successful!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo -e "${YELLOW}Use the following command to activate the environment:${NC}"
    echo -e "  ${GREEN}conda activate ${ENV_NAME}${NC}"
else
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Environment setup failed, please check error messages${NC}"
    echo -e "${RED}======================================${NC}"
    exit 1
fi

