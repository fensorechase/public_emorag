# config.py
"""
Configuration settings for the LiveRAG challenge
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Make sure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model settings
LLM_MODEL_ID = "tiiuae/falcon3-10b-instruct"  # As required by LiveRAG challenge... or "tiiuae/falcon3-10b-instruct"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
MAX_NEW_TOKENS = 1024 # Was 512
TEMPERATURE = 0.6 # Changed from 0.1
TOP_P = 0.9
DO_SAMPLE = True

# Retrieval settings
DEFAULT_TOP_K_RETRIEVAL = 1000  # For reranking
DEFAULT_TOP_K_FINAL = 10        # For final context
BM25_INDEX_PATH = "/local/scratch/kdhcfen/fineweb-index/"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model for dense retrieval
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # Lightweight reranker

# Data file paths
FINEWEB_SUBSET_PATH = os.path.join(DATA_DIR, "fineweb_subset.parquet")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "fineweb_faiss.index")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "fineweb_embeddings.npy")
METADATA_PATH = os.path.join(MODELS_DIR, "fineweb_metadata.npz")

# DataMorgana settings
DATAMORGANA_CONFIG_PATH = os.path.join(BASE_DIR, "datamorgana_config.json")
SYNTHETIC_TRAIN_PATH = os.path.join(DATA_DIR, "synthetic_train.jsonl")
SYNTHETIC_TEST_PATH = os.path.join(DATA_DIR, "synthetic_test.jsonl")

# Evaluation settings
RELEVANCE_THRESHOLD = 1.0  # Minimum score to consider an answer relevant
FAITHFULNESS_THRESHOLD = 0.5  # Minimum score to consider an answer faithful

# Latency settings
MAX_RESPONSE_TIME = 60.0  # Maximum allowed response time in seconds. Old: 10.0
EARLY_STOPPING_CONFIDENCE = 0.85  # Confidence threshold for early stopping

# Cache settings
CACHE_SIZE = 1000  # Number of items to cache
CACHE_EXPIRY = 3600  # Cache expiry time in seconds (1 hour)

# Logging settings
LOG_LEVEL = "INFO"
ENABLE_TELEMETRY = True