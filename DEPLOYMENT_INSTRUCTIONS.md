# Deployment Instructions

## Issue Identified

The error you're seeing:
```
ERROR - Failed to initialize PDF QA Engine: 'n_gpu_layers'
```

This indicates that the updated code hasn't been deployed to your Linux system at `/home/aesplai/TMI_withOSS/`.

## Solution

You need to copy the updated files from this Windows directory to your Linux system.

### Files That Were Modified

The following files were updated to implement gpt-oss-20b:

1. **pdf_qa_engine.py** - Core QA engine with gpt-oss-20b integration
2. **model_config.py** - Model configuration
3. **requirements.txt** - Added accelerate dependency
4. **start_app.sh** - Updated startup script
5. **IMPLEMENTATION_SUMMARY.md** - Documentation (new file)

### Option 1: Copy Files to Linux System

```bash
# On your Windows machine, copy files to Linux
scp pdf_qa_engine.py user@linux-host:/home/aesplai/TMI_withOSS/
scp model_config.py user@linux-host:/home/aesplai/TMI_withOSS/
scp requirements.txt user@linux-host:/home/aesplai/TMI_withOSS/
scp start_app.sh user@linux-host:/home/aesplai/TMI_withOSS/
scp IMPLEMENTATION_SUMMARY.md user@linux-host:/home/aesplai/TMI_withOSS/
```

### Option 2: Use Git (If Repository)

```bash
# On Windows (in this directory)
git add .
git commit -m "Implement gpt-oss-20b with Transformers"
git push

# On Linux
cd /home/aesplai/TMI_withOSS/
git pull
```

### Option 3: Manual Copy via Network Share or USB

1. Copy the entire `TMI_withOSS` folder to a USB drive or network share
2. Transfer to your Linux system
3. Replace the existing files

## After Copying Files

### 1. Install Updated Dependencies

```bash
cd /home/aesplai/TMI_withOSS/
source venv/bin/activate

# Install accelerate (new dependency)
pip install accelerate==0.25.0

# Verify all dependencies
pip install -r requirements.txt
```

### 2. Verify Model Configuration

```bash
python model_config.py
```

Expected output:
```
================================================================================
PDF Q&A SYSTEM - MODEL CONFIGURATION
================================================================================
Model Path: openai/gpt-oss-20b
Model Type: gpt-oss
Context Window: 4096 tokens
Device Mapping: auto
Estimated GPU Layers: 15
Estimated VRAM: 2.25 GB
CPU Threads: 8
Max Answer Length: 2048 tokens
================================================================================
Model will be downloaded from HuggingFace: openai/gpt-oss-20b
First run will download ~16GB model files

✅ Model file found!
```

### 3. Run the Application

```bash
# Using startup script
bash start_app.sh

# Or directly
python app_pdf_qa.py
```

### 4. First Run - Model Download

On first run, the gpt-oss-20b model will download automatically (~16GB):
```
Loading gpt-oss-20b model (this may take 1-2 minutes)...
Model will be downloaded from HuggingFace on first run (~16GB)
Downloading (…)lve/main/config.json: 100%|██████████| 665/665 [00:00<00:00, ...]
Downloading pytorch_model.bin: 100%|██████████| 16.2G/16.2G [05:23<00:00, ...]
```

Download time: 5-10 minutes (depending on internet speed)

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'accelerate'

```bash
pip install accelerate==0.25.0
```

### Error: Model download fails

Pre-download manually:
```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('openai/gpt-oss-20b'); \
AutoModelForCausalLM.from_pretrained('openai/gpt-oss-20b', \
device_map='auto', torch_dtype='float16')"
```

### Error: CUDA out of memory

Edit model_config.py and reduce:
```python
"n_gpu_layers": 10  # Reduce from 15 to 10
```

Or use CPU-only:
```python
"device": "cpu"
```

## Quick File Transfer Script

Create this script on Windows (`transfer_to_linux.sh`):

```bash
#!/bin/bash
# Update these with your Linux system details
LINUX_USER="aesplai"
LINUX_HOST="your-linux-hostname"
LINUX_PATH="/home/aesplai/TMI_withOSS/"

# Core files
scp pdf_qa_engine.py ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}
scp model_config.py ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}
scp requirements.txt ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}
scp start_app.sh ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}
scp app_pdf_qa.py ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}

# Documentation
scp IMPLEMENTATION_SUMMARY.md ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}
scp DEPLOYMENT_INSTRUCTIONS.md ${LINUX_USER}@${LINUX_HOST}:${LINUX_PATH}

echo "Files transferred successfully!"
echo "Now run on Linux:"
echo "  cd ${LINUX_PATH}"
echo "  source venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  python app_pdf_qa.py"
```

Run it:
```bash
bash transfer_to_linux.sh
```

## Verification Checklist

After copying files, verify:

- [ ] pdf_qa_engine.py contains `AutoModelForCausalLM` (not `GPT2LMHeadModel`)
- [ ] model_config.py has `"model_path": "openai/gpt-oss-20b"`
- [ ] requirements.txt includes `accelerate==0.25.0`
- [ ] `python model_config.py` runs without errors
- [ ] `pip list | grep accelerate` shows accelerate is installed
- [ ] `python app_pdf_qa.py` starts without the 'n_gpu_layers' error

## Summary

The implementation is complete on the Windows side. You just need to:

1. **Copy files** to your Linux system
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the app**: `python app_pdf_qa.py`

The gpt-oss-20b model will download automatically on first run.

---

**If you need help with any of these steps, let me know!**
