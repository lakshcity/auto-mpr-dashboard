import time
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


# Paths

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]

PDF_DIR = BASE_DIR / "data" / "pdfs"
INDEX_PATH = BASE_DIR / "data" / "pdf_index.faiss"
META_PATH = BASE_DIR / "data" / "pdf_meta.pkl"



# PDF Loader

def load_pdfs(pdf_dir: Path):
    docs = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if text.strip():
            docs.append({
                "source": pdf_file.name,
                "text": text
            })

    if not docs:
        raise RuntimeError("No valid PDFs found")

    return docs

# =============================
# Index Builder
# =============================
def build_pdf_index():
    print("=== PDF INDEX BUILD STARTED ===")
    start = time.time()

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF folder not found: {PDF_DIR}")

    print("Loading PDFs...")
    docs = load_pdfs(PDF_DIR)
    texts = [d["text"] for d in docs]

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding PDFs...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f"\n✅ PDF index built successfully")
    print(f"Saved: {INDEX_PATH}")
    print(f"Saved: {META_PATH}")
    print(f"⏱️ Time: {round(time.time() - start, 2)} seconds")



# Entry Point

if __name__ == "__main__":
    build_pdf_index()
