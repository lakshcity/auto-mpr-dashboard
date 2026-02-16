import os
from pathlib import Path

# ---------------------------
# Base paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# PDFs live in root data/pdfs
PDF_DIR = DATA_DIR / "pdfs"

# ---------------------------
# Vector store paths
# ---------------------------
PDF_INDEX = DATA_DIR / "pdf_index.faiss"
PDF_META = DATA_DIR / "pdf_meta.pkl"
PDF_REGISTRY = DATA_DIR / "index_registry.json"
# ---------------------------
# Models & Retrieval config
# ---------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# LLM (generation only, NOT embeddings)
OLLAMA_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")

TOP_K = int(os.getenv("TOP_K", "6"))

# ---------------------------
# Safety logs (optional but useful)
# ---------------------------
print(f"[CONFIG] EMBED_MODEL = {EMBED_MODEL}")
print(f"[CONFIG] OLLAMA_MODEL = {OLLAMA_MODEL}")
print(f"[CONFIG] TOP_K = {TOP_K}")

