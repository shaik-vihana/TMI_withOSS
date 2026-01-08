#!/bin/bash

# =============================================================================
# PDF Chat with Mistral 7B - Setup & Cleanup Script
# =============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "PDF CHAT SYSTEM SETUP (Mistral 7B + Ollama)"
echo "================================================================================"

# 1. Cleanup Old GPT-OSS Files
echo "[1/5] Cleaning up old GPT-OSS files..."
rm -f INSTALL_GPT_OSS.md GPT_OSS_SUMMARY.md IMPLEMENTATION_SUMMARY.md DEPLOYMENT_INSTRUCTIONS.md DEPLOYMENT_CHECKLIST.md
rm -rf static templates  # Streamlit doesn't use these
rm -rf chroma_db         # Switched to FAISS
rm -rf logs              # Clear old logs
echo "✓ Old documentation and assets removed"

# 2. Environment Setup
echo
echo "[2/5] Setting up Python environment..."

if [ -d "venv" ]; then
    echo "Virtual environment exists."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

source venv/bin/activate

# 3. Install Dependencies
echo
echo "[3/5] Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "✓ All dependencies installed"

# 4. Check Ollama & Mistral
echo
echo "[4/5] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "⚠ Ollama not found. Please install from https://ollama.com"
else
    echo "✓ Ollama detected"
    if ! ollama list | grep -q "mistral"; then
        echo "Pulling Mistral 7B model..."
        ollama pull mistral
    else
        echo "✓ Mistral model already present"
    fi
fi

# 5. Launch
echo
echo "[5/5] Setup Complete!"
echo "================================================================================"
echo "To start the app:"
echo "  streamlit run app_pdf_qa.py"
echo "================================================================================"
