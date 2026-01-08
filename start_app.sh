#!/bin/bash
# PDF Q&A System with OpenAI gpt-oss-20b - Startup Script
# Updated for gpt-oss-20b model

set -e

echo "========================================================================="
echo "   PDF Q&A SYSTEM WITH GPT-OSS-20B"
echo "   Powered by OpenAI's gpt-oss-20b (21B params, 3.6B active, Apache 2.0)"
echo "========================================================================="
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
NEED_INSTALL=false

# Check if transformers is installed and recent enough (>=4.46.0)
if python -c "import transformers" 2>/dev/null; then
    VER=$(python -c "import transformers; print(transformers.__version__)")
    # Check if version starts with 4.3 or lower (simple check)
    if [[ "$VER" == 4.3* ]] || [[ "$VER" == 4.2* ]] || [[ "$VER" == 3.* ]]; then
        echo "Found transformers $VER, but >= 4.46.0 is required."
        NEED_INSTALL=true
    fi
else
    NEED_INSTALL=true
fi

if [ "$NEED_INSTALL" = true ] || ! python -c "import flask" 2>/dev/null || ! python -c "import fitz" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install --upgrade pip --quiet

    # Install remaining dependencies
    echo "Installing remaining packages..."
    pip install -r requirements.txt --quiet

    echo -e "${GREEN}Dependencies installed successfully${NC}"
else
    echo "Dependencies already installed"
fi
echo

# Check for GPU (optional)
echo "[5/7] Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "GPU will be used for model inference (faster response times)"
else
    echo "No NVIDIA GPU detected. Using CPU-only mode."
    echo "Response times may be slower (30-60s instead of 5-15s)"
fi
echo

# Check if model exists
echo "[6/7] Checking for gpt-oss-20b model..."
MODEL_PATH=$(python -c "from model_config import get_model_path; print(get_model_path())" 2>/dev/null || echo "openai/gpt-oss-20b")

# Check if model path is a HuggingFace model ID (contains '/')
if [[ "$MODEL_PATH" == *"/"* ]]; then
    echo -e "${GREEN}Model will be auto-downloaded from HuggingFace: $MODEL_PATH${NC}"
    echo "Model will be downloaded on first run (~16GB)"
    echo "Download location: ~/.cache/huggingface/hub/"
    echo
    echo "To pre-download the model now (optional), you can run:"
    echo "  python -c \"from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL_PATH'); AutoModelForCausalLM.from_pretrained('$MODEL_PATH', device_map='auto', torch_dtype='float16')\""
    echo
elif [ ! -d "$MODEL_PATH" ]; then
    echo
    echo "========================================================================"
    echo -e "${YELLOW}WARNING: gpt-oss-20b model not found${NC}"
    echo "========================================================================"
    echo
    echo "Model path: $MODEL_PATH"
    echo
    echo "The model will be downloaded automatically from HuggingFace on first run."
    echo "Alternatively, you can set MODEL_PATH=openai/gpt-oss-20b in your environment."
    echo
    echo "Download size: ~16GB"
    echo "See INSTALL_GPT_OSS.md for detailed instructions"
    echo
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
