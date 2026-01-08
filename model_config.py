"""
Model Configuration for PDF Q&A System with 20B LLM
Supports CPU + GPU offloading for efficient inference
"""

import os
from pathlib import Path

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Model identifier (HuggingFace model name or local path)
    # OpenAI gpt-oss-20b: 21B total params, 3.6B active (MoE)
    # Will be downloaded automatically from HuggingFace on first run
    "model_path": os.getenv("MODEL_PATH", "openai/gpt-oss-20b"),

    # Model type
    "model_type": "gpt-oss",  # OpenAI gpt-oss architecture (MoE)

    # Context window
    "n_ctx": 4096,  # gpt-oss-20b context window

    # GPU offloading (automatic with device_map="auto")
    # Transformers will automatically distribute layers between CPU and GPU
    "device": "auto",  # auto, cpu, cuda

    # CPU threads for CPU layers (when using mixed CPU+GPU)
    "n_threads": 8,  # Recommended: number of CPU cores / 2

    # Generation settings
    "max_tokens": 2048,  # Maximum answer length
    "temperature": 0.3,  # Lower = more focused, higher = more creative (0.0-1.0)
    "top_p": 0.95,  # Nucleus sampling
    "top_k": 40,  # Top-k sampling

    # GPU layers estimate (for reference - actual handled by device_map="auto")
    "n_gpu_layers": 15,  # Estimated layers on GPU for 4GB VRAM
}

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================

INFERENCE_CONFIG = {
    # Response time target (seconds)
    "target_response_time": 15,  # 5-15 seconds acceptable

    # Confidence score calculation
    "use_confidence_scoring": True,
    "confidence_method": "perplexity",  # Method: perplexity, logprobs, similarity

    # Page number tracking
    "track_page_numbers": True,
    "multi_page_support": True,  # Support answers from multiple pages
    "max_pages_per_answer": 5,  # Maximum pages to cite in one answer
}

# ============================================================================
# PDF PROCESSING SETTINGS
# ============================================================================

PDF_CONFIG = {
    # DPI for page rendering
    "dpi": 150,  # Options: 100, 150, 200, 300

    # PDF size limits
    "max_file_size_mb": 50,  # Maximum PDF size
    "max_pages": 1000,  # Maximum number of pages

    # Text extraction
    "extract_text": True,
    "extract_images": True,

    # Chunking strategy
    "chunk_size": 512,  # Tokens per chunk
    "chunk_overlap": 50,  # Overlap between chunks
}

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

RETRIEVAL_CONFIG = {
    # Number of pages to retrieve
    "top_k": 5,

    # Retrieval method
    "use_semantic_search": True,  # Use embeddings for search
    "use_keyword_search": True,  # Also use keyword matching

    # Embedding model
    "embedding_model": "all-MiniLM-L6-v2",  # Fast and accurate

    # Vector store
    "vector_store": "chroma",  # Options: chroma, faiss
    "persist_directory": "chroma_db",
}

# ============================================================================
# ANALYTICS & LOGGING
# ============================================================================

ANALYTICS_CONFIG = {
    # Log all Q&A sessions
    "enable_logging": True,
    "log_file": "logs/qa_performance.txt",

    # Track metrics
    "track_response_time": True,
    "track_confidence": True,
    "track_page_references": True,

    # Session management
    "session_timeout": 3600,  # 1 hour
    "max_sessions": 100,  # Maximum concurrent sessions
}

# ============================================================================
# SYSTEM REQUIREMENTS
# ============================================================================

SYSTEM_REQUIREMENTS = {
    "ram_gb": 40,
    "gpu_vram_gb": 4,
    "storage_gb": 20,
    "os": "Ubuntu 24",

    # Estimated memory usage (gpt-oss-20b with MXFP4 quantization)
    # MoE architecture: 21B total params, but only 3.6B active per token
    "model_ram_usage_gb": 16,  # Model loaded in RAM (MXFP4 quantized)
    "model_vram_usage_gb": 2.5,  # 15 layers on GPU (~2.5GB)
    "total_ram_needed_gb": 20,  # Model + ChromaDB + Flask + overhead
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path():
    """Get model path (HuggingFace model ID or absolute local path)."""
    model_path_str = MODEL_CONFIG["model_path"]

    # If it's a HuggingFace model ID (contains '/'), return as-is
    if '/' in model_path_str and not model_path_str.startswith('/'):
        return model_path_str

    # Otherwise, treat as local path and make it absolute
    model_path = Path(model_path_str)

    if not model_path.is_absolute():
        # If relative path, make it relative to this config file
        config_dir = Path(__file__).parent
        model_path = config_dir / model_path

    return str(model_path)

def validate_model_exists():
    """Check if model is available (local or will be downloaded from HuggingFace)."""
    model_path = get_model_path()

    # If it's a HuggingFace model ID (contains '/'), it will be auto-downloaded
    if '/' in model_path:
        print(f"Model will be downloaded from HuggingFace: {model_path}")
        print("First run will download ~16GB model files")
        return True

    # If it's a local path, check if it exists
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n\n"
            f"Please download the gpt-oss-20b model:\n\n"
            f"Option 1: Use HuggingFace model ID (recommended)\n"
            f"  Set MODEL_PATH=openai/gpt-oss-20b\n"
            f"  Model will be downloaded automatically\n\n"
            f"Option 2: Download manually\n"
            f"  Visit: https://huggingface.co/openai/gpt-oss-20b\n"
            f"  Download the model files and place in local directory\n\n"
            f"See INSTALL_GPT_OSS.md for detailed instructions"
        )

    # Check for required files in local directory
    required_files = ['config.json']
    missing_files = [f for f in required_files if not (Path(model_path) / f).exists()]

    if missing_files:
        raise FileNotFoundError(
            f"Model directory found but missing required files: {missing_files}\n"
            f"Model path: {model_path}\n\n"
            f"Please re-download the gpt-oss-20b model.\n\n"
            f"See INSTALL_GPT_OSS.md for detailed instructions"
        )

    return True

def estimate_vram_usage():
    """Estimate VRAM usage based on n_gpu_layers."""
    layers = MODEL_CONFIG["n_gpu_layers"]

    # Rough estimate: ~150MB per layer for 20B model
    vram_gb = (layers * 150) / 1024

    return round(vram_gb, 2)

def get_recommended_layers_for_vram(vram_gb):
    """Get recommended n_gpu_layers for given VRAM."""
    if vram_gb <= 2:
        return 0  # CPU only
    elif vram_gb <= 4:
        return 12  # ~2GB VRAM usage
    elif vram_gb <= 6:
        return 20  # ~3GB VRAM usage
    elif vram_gb <= 8:
        return 35  # ~5GB VRAM usage
    else:
        return 60  # Full offload for 12GB+ VRAM

# ============================================================================
# PRINT CONFIGURATION
# ============================================================================

def print_config():
    """Print current configuration."""
    print("="*80)
    print("PDF Q&A SYSTEM - MODEL CONFIGURATION")
    print("="*80)
    print(f"Model Path: {get_model_path()}")
    print(f"Model Type: {MODEL_CONFIG['model_type']}")
    print(f"Context Window: {MODEL_CONFIG['n_ctx']} tokens")
    print(f"Device Mapping: {MODEL_CONFIG['device']}")
    print(f"Estimated GPU Layers: {MODEL_CONFIG.get('n_gpu_layers', 'N/A')}")
    print(f"Estimated VRAM: {estimate_vram_usage()} GB")
    print(f"CPU Threads: {MODEL_CONFIG.get('n_threads', 'N/A')}")
    print(f"Max Answer Length: {MODEL_CONFIG['max_tokens']} tokens")
    print("="*80)
    print(f"Target Response Time: {INFERENCE_CONFIG['target_response_time']}s")
    print(f"Confidence Scoring: {INFERENCE_CONFIG['use_confidence_scoring']}")
    print(f"Multi-Page Support: {INFERENCE_CONFIG['multi_page_support']}")
    print("="*80)

if __name__ == "__main__":
    print_config()

    try:
        validate_model_exists()
        print("\n✅ Model file found!")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
