import pickle
import faiss
from sentence_transformers import SentenceTransformer

from core.config import PDF_INDEX, PDF_META, EMBED_MODEL, TOP_K

# ---------------------------
# Load Resources
# ---------------------------
MODEL = SentenceTransformer(EMBED_MODEL)

INDEX = faiss.read_index(str(PDF_INDEX))

with open(PDF_META, "rb") as f:
    METADATA = pickle.load(f)

print("Retriever loaded ✅")


# ---------------------------
# RAG Context Retrieval
# ---------------------------
def retrieve_context(query):

    if not query.strip():
        return ""

    query_vec = MODEL.encode([query]).astype("float32")
    _, indices = INDEX.search(query_vec, TOP_K)

    chunks = []

    for i in indices[0]:
        if i < len(METADATA):
            text = METADATA[i].get("text", "")
            if text:
                chunks.append(text)

    return "\n".join(chunks)


def find_similar_cases(query, top_k=5):

    if not query.strip():
        return []

    query_vec = MODEL.encode([query]).astype("float32")
    distances, indices = INDEX.search(query_vec, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0 or idx >= len(METADATA):
            continue

        case = METADATA[idx].copy()

        confidence = max(0, 100 - float(distances[0][rank]))
        case["confidence"] = round(confidence, 2)

        results.append(case)

    return results


def format_context(context_text):

    if not context_text:
        return ""

    return f"\n\n📄 Context:\n{context_text}"
