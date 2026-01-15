import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "data" / "case_index.faiss"
META_PATH  = BASE_DIR / "data" / "case_meta.pkl"

# Load once
print("Loading search engine...")
index = faiss.read_index(str(INDEX_PATH))

with open(META_PATH, "rb") as f:
    cases = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def find_similar_cases(query, top_k=5):
    """
    Returns top_k most similar historical MPRs
    """
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        case = cases[idx]
        score = float(distances[0][i])
        results.append({
            "similarity_score": score,
            "case": case
        })

    return results
