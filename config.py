# Configuration Module
"""
Configuration settings for AI Academic Assistant
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTORSTORE_FOLDER = os.path.join(BASE_DIR, "vectorstore")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# ==================== EMBEDDING CONFIG ====================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, lightweight, good quality
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DIMENSION = 384  # For all-MiniLM-L6-v2

# ==================== LLM CONFIG ====================
LLM_TYPE = os.getenv("LLM_TYPE", "ollama")  # "ollama" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")  # Using tinyllama for stability
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_MAX_TOKENS = 2048

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ==================== RAG CONFIG ====================
RAG_TOP_K = 5  # Number of documents to retrieve
RAG_OVERLAP = 100  # Character overlap in chunks
RAG_CHUNK_SIZE = 512  # Size of text chunks

# ==================== STREAMLIT CONFIG ====================
STREAMLIT_THEME = "light"
STREAMLIT_PAGE_ICON = "📚"
STREAMLIT_LAYOUT = "wide"

# ==================== LOGGING CONFIG ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== FILE UPLOAD CONFIG ====================
MAX_FILE_SIZE_MB = 50
ALLOWED_FILE_TYPES = [".pdf"]
