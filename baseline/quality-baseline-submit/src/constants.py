# Common constants for the quality predictor pipeline

import os

# Presets are specific to Qwen3VL models
PIXEL_PRESETS = {
    "S": 100352,
    "M": 261120,
    "L": 401408,
}

# Default inference batch sizes tuned for H100 80GB (override via CLI or env)
DEFAULT_EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "128"))
DEFAULT_LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "64"))
DEFAULT_PIXEL_PRESET = os.environ.get("PIXEL_PRESET", "M").upper()

# Classifier artifact path (relative to submit root)
_ARTIFACT_REL = "artifacts/product_quality_predictor.joblib"

# Models directory (matches evaluator's SHARED_MODELS_PATH convention)
SHARED_MODELS_DIR = "SHARED_MODELS_PATH"
DEFAULT_MODELS_DIR = "/shared_models"

# Embedding model path
EMBED_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"

# LLM model path
LLM_MODEL_NAME = "Qwen/Qwen3.5-4B"

# Comment formatting constants
MIN_COMMENT_LEN = 50
MAX_COMMENT_LEN = 300
