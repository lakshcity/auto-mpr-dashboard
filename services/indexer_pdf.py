import json
import sys
import hashlib
import pickle
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- FIX: Add project root to system path ---
# This allows python to find 'core' even when running from 'services/'
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
# --------------------------------------------

from core.config import (
    PDF_DIR,
    PDF_INDEX,
    PDF_META,
    PDF_REGISTRY,
    EMBED_MODEL
)

# ---------------------------
# Load Embedding Model
# ---------------------------
model = SentenceTransformer(EMBED_MODEL)
print(f"Embedding model loaded: {EMBED_MODEL}")
  # should be all-MiniLM-L6-v2


# ---------------------------
# Helpers
# ---------------------------
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry():
    if PDF_REGISTRY.exists():
        return json.loads(PDF_REGISTRY.read_text())
    return {"indexed_files": {}}


def save_registry(registry):
    PDF_REGISTRY.write_text(json.dumps(registry, indent=2))


def extract_text_chunks(pdf_path: Path, chunk_size=500):
    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        text = page.get_text()
        if text:
            full_text.append(text)

    words = " ".join(full_text).split()

    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
        if len(words[i:i + chunk_size]) > 20
    ]


# ---------------------------
# Incremental Index Builder
# ---------------------------
def incremental_index():

    registry = load_registry()
    indexed_files = registry["indexed_files"]

    new_files = []

    for pdf in PDF_DIR.glob("*.pdf"):
        h = file_hash(pdf)

        if pdf.name not in indexed_files or indexed_files[pdf.name]["hash"] != h:
            new_files.append((pdf, h))

    if not new_files:
        print("No new or modified PDFs found")
        return

    # Load or create index
    if PDF_INDEX.exists() and PDF_META.exists():
        index = faiss.read_index(str(PDF_INDEX))
        with open(PDF_META, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(384)  # MiniLM = 384 dims
        metadata = []

    for pdf, h in new_files:

        print(f"Processing PDF: {pdf.name}")

        chunks = extract_text_chunks(pdf)

        if not chunks:
            print(f"⚠️ No text extracted from {pdf.name}")
            continue

        vectors = model.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        ).astype("float32")

        index.add(vectors)

        for chunk in chunks:
            metadata.append({
                "text": chunk,
                "source": pdf.name
            })

        registry["indexed_files"][pdf.name] = {
            "hash": h,
            "indexed_at": datetime.utcnow().isoformat()
        }

    # Save artifacts
    faiss.write_index(index, str(PDF_INDEX))
    with open(PDF_META, "wb") as f:
        pickle.dump(metadata, f)

    save_registry(registry)

    print("✅ PDF indexing completed successfully")
    print(f"FAISS index size: {index.ntotal}")
    print(f"FAISS dimension: {index.d}")


# ---------------------------
# Run directly
# ---------------------------
if __name__ == "__main__":
    incremental_index()
