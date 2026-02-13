import time
import pickle
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parents[1]
PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_PATH = BASE_DIR / "data" / "pdf_index.faiss"
META_PATH = BASE_DIR / "data" / "pdf_meta.pkl"

CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 100


# -----------------------------------
# Text Chunking
# -----------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) > 50:  # avoid tiny fragments
            chunks.append(" ".join(chunk))

    return chunks


# -----------------------------------
# PDF Loader with Chunking
# -----------------------------------
def load_and_chunk_pdfs(pdf_dir: Path):
    docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        full_text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        if full_text.strip():
            chunks = chunk_text(full_text)

            for idx, chunk in enumerate(chunks):
                docs.append({
                    "source": pdf_file.name,
                    "chunk_id": idx,
                    "text": chunk
                })

    if not docs:
        raise RuntimeError("No valid PDFs found")

    return docs


# -----------------------------------
# Index Builder
# -----------------------------------
def build_pdf_index():
    print("=== CHUNKED PDF INDEX BUILD STARTED ===")
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

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"✅ Chunked PDF index built")
    print(f"Total chunks: {len(docs)}")
    print(f"⏱ Time: {round(time.time() - start, 2)} sec")


if __name__ == "__main__":
    build_pdf_index()
