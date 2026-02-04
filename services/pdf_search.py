import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]

INDEX_PATH = BASE_DIR / "data" / "pdf_index.faiss"
META_PATH = BASE_DIR / "data" / "pdf_meta.pkl"

# Load once (IMPORTANT)
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)


def search_pdf(query: str, top_k: int = 3):
    if not query.strip():
        return []

    query_vec = model.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)

    return [metadata[i] for i in indices[0]]
