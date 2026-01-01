#!/bin/bash
# PDF Q&A System with OpenAI gpt-oss-20b - Startup Script
# Updated for gpt-oss-20b model

set -e

echo "========================================================================"
echo "   PDF Q&A SYSTEM WITH GPT-OSS-20B"
echo "   Powered by OpenAI's gpt-oss-20b (21B params, Apache 2.0)"
echo "========================================================================"
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo "Please install Python 3.9+ from https://www.python.org"
    exit 1
fi

echo "[1/7] Checking Python installation..."
python3 --version
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[2/7] Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created successfully${NC}"
else
    echo "[2/7] Virtual environment already exists"
fi
echo

# Activate virtual environment
echo "[3/7] Activating virtual environment..."
source venv/bin/activate
echo

# Check if dependencies are installed
echo "[4/7] Checking dependencies..."

# Quick check for key packages
if ! python -c "import flask" 2>/dev/null || ! python -c "import fitz" 2>/dev/null || ! python -c "import tqdm" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip --quiet

    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Installing llama-cpp-python with CUDA support..."
        CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir --quiet
    else
        echo "No NVIDIA GPU detected. Installing CPU-only version..."
        pip install llama-cpp-python --quiet
    fi

    # Install remaining dependencies
    echo "Installing remaining packages..."
    pip install -r requirements.txt --quiet

    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo "Dependencies already installed"
fi
echo

# Check if Ollama is installed (optional, for easy model download)
echo "[5/7] Checking for Ollama (optional)..."
if command -v ollama &> /dev/null; then
    echo "Ollama found: $(ollama --version)"
    OLLAMA_AVAILABLE=true
else
    echo "Ollama not found (optional - can download model manually)"
    OLLAMA_AVAILABLE=false
fi
echo

# Check if model exists
echo "[6/7] Checking for GPT-2 OSS model..."
MODEL_PATH=$(python -c "from model_config import get_model_path; print(get_model_path())" 2>/dev/null || echo "./gpt2-oss")

if [ ! -d "$MODEL_PATH" ]; then
    echo
    echo "========================================================================"
    echo -e "${YELLOW}WARNING: GPT-2 OSS model not found${NC}"
    echo "========================================================================"
    echo
    echo "Model path: $MODEL_PATH"
    echo

    echo "Downloading GPT-2 OSS model using transformers..."
    echo
    python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
print('Downloading GPT-2 OSS model...')
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2-medium')
print('Saving model locally...')
tokenizer.save_pretrained('./gpt2-oss')
model.save_pretrained('./gpt2-oss')
print('Model downloaded successfully!')
"

    if [ $? -eq 0 ]; then
        echo
        echo -e "${GREEN}Model downloaded successfully!${NC}"
        echo
    else
        echo
        echo -e "${RED}Failed to download model${NC}"
        echo "Please download manually from https://huggingface.co/openai-community/gpt2-medium"
        exit 1
    fi
else
    echo -e "${GREEN}Model found: $MODEL_PATH${NC}"
fi
echo

# Verify model configuration
echo "[7/7] Verifying model configuration..."
if python model_config.py 2>/dev/null; then
    echo -e "${GREEN}Configuration verified${NC}"
else
    echo -e "${YELLOW}Warning: Could not verify configuration${NC}"
    echo "Application may fail to start"
fi
echo

echo "========================================================================"
echo "   STARTING APPLICATION"
echo "========================================================================"
echo
echo "Server will start at: http://localhost:5000"
echo "Analytics dashboard: http://localhost:5000/view-log"
echo "Health check: http://localhost:5000/health"
echo
echo "Press Ctrl+C to stop the server"
echo
echo "========================================================================"
echo

# Start the application
python app_pdf_qa.py

# Cleanup on exit
trap "echo 'Shutting down...'; deactivate; exit" INT TERM
