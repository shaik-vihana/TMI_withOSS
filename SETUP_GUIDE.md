# Quick Setup Guide - PDF Q&A System with 20B LLM

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Run automated setup
chmod +x setup.sh
./setup.sh
```

The setup script will:
- ✅   Check Python version
- ✅ Create virtual environment
- ✅ Install llama-cpp-python with CUDA support (if GPU available)
- ✅ Install all dependencies
- ✅ Optionally download a model

### Step 2: Download Model (If Not Done in Step 1)

**Recommended**: Qwen2.5-14B-Instruct-Q4_K_M.gguf (~8GB)

```bash
# Download from HuggingFace
wget https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct.Q4_K_M.gguf \
  -O models/qwen2.5-14b-instruct.Q4_K_M.gguf
```

### Step 3: Configure Model Path

Edit [model_config.py](model_config.py:8-9):

```python
MODEL_CONFIG = {
    "model_path": "models/qwen2.5-14b-instruct.Q4_K_M.gguf",
    # ...
}
```

### Step 4: Adjust GPU Layers (Optional)

For 4GB VRAM, keep default (12 layers).

For different VRAM, edit [model_config.py](model_config.py:17-24):

```python
MODEL_CONFIG = {
    "n_gpu_layers": 12,  # Adjust based on your VRAM
    #  0: CPU only
    # 10-15: 4GB VRAM (recommended: 12)
    # 20-25: 6GB VRAM
    # 35+: 8GB+ VRAM
}
```

### Step 5: Start Application

```bash
# Activate virtual environment
source venv/bin/activate

# Start app
python app_pdf_qa.py
```

### Step 6: Access Application

Open browser: **http://localhost:5000**

---

## 📁 File Structure Overview

```
TMI_withOSS/
├── app_pdf_qa.py              # Main Flask app (start here!)
├── pdf_qa_engine.py           # QA engine with 20B LLM
├── pdf_processor.py           # PDF text/image extraction
├── model_config.py            # ⚙️ Configure model & GPU here
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
├── models/                    # Put your GGUF models here
└── templates/                 # HTML UI files
```

---

## 🔧 Configuration Cheat Sheet

### GPU Offloading

| VRAM | `n_gpu_layers` | Performance |
|------|----------------|-------------|
| None | 0 | 5-10 tok/s (CPU only) |
| 4GB | 10-15 | 20-30 tok/s |
| 6GB | 20-25 | 30-40 tok/s |
| 8GB+ | 35+ | 40-60 tok/s |

### Model Selection

| Model | Size | RAM Needed | Quality |
|-------|------|------------|---------|
| Mistral-7B-Q4_K_M | 4GB | 8GB | Good |
| Qwen2.5-14B-Q4_K_M | 8GB | 16GB | **Excellent** ⭐ |
| Mixtral-8x7B-Q4_K_M | 26GB | 32GB | Superior |

---

## 🐛 Quick Troubleshooting

### Problem: "Model file not found"

```bash
# Download model
wget https://huggingface.co/TheBloke/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct.Q4_K_M.gguf \
  -O models/qwen2.5-14b-instruct.Q4_K_M.gguf

# Update model_config.py
nano model_config.py
# Set: "model_path": "models/qwen2.5-14b-instruct.Q4_K_M.gguf"
```

### Problem: "CUDA out of memory"

```python
# Edit model_config.py - Reduce GPU layers
"n_gpu_layers": 8,  # Try 8 instead of 12
```

### Problem: llama-cpp-python install fails

```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential cmake

# Reinstall with CUDA
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Problem: Slow inference (CPU only)

```bash
# Reinstall with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Edit model_config.py
"n_gpu_layers": 12,  # Enable GPU offloading
```

---

## 📊 What to Expect

### Upload & Processing
- **100-page PDF**: 30-60 seconds
- Creates searchable index of all pages

### Question Answering
- **With GPU (12 layers)**: 5-15 seconds per answer
- **CPU only**: 30-90 seconds per answer
- Includes exact page references and confidence score

### Example Output

```
Q: What is the revenue growth strategy?

A: The revenue growth strategy focuses on three key areas:
   1. Market expansion into Asia-Pacific (Page 5)
   2. Launch of new product lines (Pages 12-14)
   3. Strategic partnerships (Page 18)

📄 Pages Referenced: 5, 12-14, 18
📊 Confidence: 87.3%
⏱️ Response Time: 8.2s
```

---

## 🎯 Key Features

✅ **Multi-Page Answers** - Cites multiple pages (e.g., "Pages 5, 12-14, 18")
✅ **Confidence Scores** - 0-100% confidence for each answer
✅ **Fast Responses** - 5-15 seconds with GPU offloading
✅ **Image Support** - Shows diagrams and charts from PDF
✅ **Analytics Dashboard** - http://localhost:5000/view-log

---

## 🚀 Next Steps

1. ✅ **Test with a PDF** - Upload a document and ask questions
2. ✅ **Check Analytics** - Visit http://localhost:5000/view-log
3. ✅ **Optimize Performance** - Adjust `n_gpu_layers` in model_config.py
4. ✅ **Try Different Models** - Download and test other GGUF models

---

## 📖 Full Documentation

See [README.md](README.md) for complete documentation including:
- Detailed configuration options
- API endpoints
- Troubleshooting guide
- Usage examples

---

**Ready to Go!** 🎉

Your PDF Q&A system is configured and ready to process documents with AI-powered question answering.
