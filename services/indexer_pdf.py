import time
import pickle
import re
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_PATH = BASE_DIR / "data" / "pdf_index.faiss"
META_PATH = BASE_DIR / "data" / "pdf_meta.pkl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 80


# ---------------------------------------------------
# Text Cleaning
# ---------------------------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    text = text.replace("\x00", "")
    return text.strip()


# ---------------------------------------------------
# Paragraph-Based Smart Chunking
# ---------------------------------------------------
def chunk_text(text, chunk_size=300, overlap=80):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []

    for para in paragraphs:
        words = para.split()

        if len(" ".join(current_chunk).split()) + len(words) <= chunk_size:
            current_chunk.append(para)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # apply overlap
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words), para]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------
# PDF Loader
# ---------------------------------------------------
def load_and_chunk_pdfs(pdf_dir: Path):
    docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        full_text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        full_text = clean_text(full_text)

        if full_text:
            chunks = chunk_text(full_text)

            for idx, chunk in enumerate(chunks):
                docs.append({
                    "source": pdf_file.name,
                    "chunk_id": idx,
                    "text": chunk,
                    "length": len(chunk.split())
                })

    if not docs:
        raise RuntimeError("No valid PDFs found")

    return docs


# ---------------------------------------------------
# Index Builder
# ---------------------------------------------------
def build_pdf_index():
    print("=== HYBRID OPTIMIZED PDF INDEX BUILD STARTED ===")
    start = time.time()

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF folder not found: {PDF_DIR}")

    print("Loading and chunking PDFs...")
    docs = load_and_chunk_pdfs(PDF_DIR)
    texts = [d["text"] for d in docs]

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    ).astype("float32")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    print("Building FAISS cosine index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    print("✅ Optimized PDF index built")
    print(f"Total chunks: {len(docs)}")
    print(f"⏱ Time: {round(time.time() - start, 2)} sec")


if __name__ == "__main__":
    build_pdf_index()
