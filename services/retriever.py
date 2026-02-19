# =====================================================
# 🚀 RETRIEVER: PDF RAG + REDASH HYBRID (COSINE OPTIMIZED)
# =====================================================

import pickle
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from core.config import PDF_INDEX, PDF_META, EMBED_MODEL, TOP_K
from services.feedback_manager import (
    get_weighted_subject_score,
    get_low_performing_subjects,
    get_feedback_stats
)




# =====================================================
# 🔹 1️⃣ LOAD PDF RAG RESOURCES
# =====================================================

print("🔄 Loading embedding model...")
MODEL = SentenceTransformer(EMBED_MODEL)

print("🔄 Loading PDF FAISS cosine index...")
INDEX = faiss.read_index(str(PDF_INDEX))

with open(PDF_META, "rb") as f:
    METADATA = pickle.load(f)

print("✅ PDF Retriever Ready")


# =====================================================
# 🔹 2️⃣ LOAD REDASH FILE
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

REDASH_DF = pd.DataFrame()
REDASH_INDEX = None


def load_redash_file():
    global REDASH_DF

    possible_files = [
        DATA_DIR / "redash_latest.xlsx",
        DATA_DIR / "redash_latest.xls",
        DATA_DIR / "redash_latest.csv"
    ]

    for file_path in possible_files:
        if file_path.exists():
            try:
                if file_path.suffix == ".csv":
                    REDASH_DF = pd.read_csv(file_path)
                else:
                    REDASH_DF = pd.read_excel(file_path)

                REDASH_DF.columns = (
                    REDASH_DF.columns
                    .str.strip()
                    .str.lower()
                )

                print(f"✅ Loaded Redash file: {file_path.name}")
                return
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")

    print("❌ No Redash file found.")


load_redash_file()


# =====================================================
# 🔹 3️⃣ REDASH SEARCH (Adaptive Ranking)
# =====================================================

def search_redash_mpr(query, top_k=5):

    global REDASH_INDEX

    if REDASH_DF.empty or not query.strip():
        return []

    subject_col = "mpr_subject"

    if subject_col not in REDASH_DF.columns:
        return []

    query_lower = query.lower()

    # ---------- KEYWORD MATCH ----------
    keyword_df = REDASH_DF[
        REDASH_DF[subject_col]
        .astype(str)
        .str.lower()
        .str.contains(query_lower, na=False)
    ]

    if not keyword_df.empty:
        results = keyword_df.head(top_k).to_dict("records")
        return _apply_adaptive_scoring(results)

    # ---------- SEMANTIC MATCH ----------
    if REDASH_INDEX is None:

        combined_text = (
            REDASH_DF["mpr_subject"].astype(str).fillna("") + " " +
            REDASH_DF.get("details", "").astype(str)
        )

        subjects = combined_text.tolist()

        embeddings = MODEL.encode(
            subjects,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype("float32")

        REDASH_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
        REDASH_INDEX.add(embeddings)

    query_vec = MODEL.encode([query]).astype("float32")
    distances, indices = REDASH_INDEX.search(query_vec, top_k)

    results = []

    for rank, idx in enumerate(indices[0]):
        row = REDASH_DF.iloc[idx].to_dict()

        max_dist = max(distances[0])
        min_dist = min(distances[0])

        if max_dist == min_dist:
            confidence = 80
        else:
            normalized = (max_dist - distances[0][rank]) / (max_dist - min_dist)
            confidence = round(normalized * 100, 2)

        row["confidence"] = confidence
        results.append(row)

    enhanced = _apply_adaptive_scoring(results)
    enhanced = apply_confidence_threshold(enhanced)
    return enhanced


def _apply_adaptive_scoring(results):

    low_subjects = get_low_performing_subjects()
    enhanced = []
    import numpy as np # Import at the top of the function for cleanliness

    for r in results:

        subject = r.get("mpr_subject", "")
        success_rate = get_weighted_subject_score(subject)
        reward_scaled = success_rate * 20

        # Calculate base scores FIRST
        final_score = (
            0.7 * r.get("confidence", 0) +
            0.3 * reward_scaled
        )

        raw_conf = (
            0.6 * r.get("confidence", 0) +
            0.4 * reward_scaled
        )

        # Calculate composite_confidence FIRST
        # Smooth extreme spikes
        composite_confidence = 100 * np.tanh(raw_conf / 100)

        # Apply penalties AFTER base calculation
        if subject in low_subjects:
            final_score *= 0.4
            composite_confidence *= 0.5
        elif success_rate < 2:
            final_score *= 0.75

        r["adaptive_score"] = round(final_score, 2)
        r["composite_confidence"] = round(composite_confidence, 2)

        enhanced.append(r)

    return enhanced


def apply_confidence_threshold(results, base_threshold=30):
    """
    Dynamically filter low-confidence results.
    Threshold increases if model stability is poor.
    """
    stats = get_feedback_stats()
    model_std = stats.get("reward_std", 0)

    # If unstable model, be stricter
    dynamic_threshold = base_threshold

    if model_std > 1.5:
        dynamic_threshold += 10
    elif model_std > 1.0:
        dynamic_threshold += 5

    filtered = [
        r for r in results
        if r.get("composite_confidence", 0) >= dynamic_threshold
    ]

    return filtered



# =====================================================
# 🔹 4️⃣ PDF CONTEXT RETRIEVAL (COSINE + KEYWORD BOOST)
# =====================================================

def retrieve_context(query, return_scores=False):

    if not query.strip():
        return "", []

    query_vec = MODEL.encode([query]).astype("float32")
    faiss.normalize_L2(query_vec)

    distances, indices = INDEX.search(query_vec, TOP_K * 2)

    ranked_chunks = []
    query_terms = query.lower().split()

    for rank, idx in enumerate(indices[0]):

        if 0 <= idx < len(METADATA):
            text = METADATA[idx].get("text", "")

            if text:

                semantic_score = distances[0][rank]

                # Convert cosine similarity to %
                confidence_percent = max(0, semantic_score) * 100

                keyword_score = sum(
                    term in text.lower() for term in query_terms
                )

                # Keyword boost scaled properly
                combined_score = confidence_percent + (5 * keyword_score)

                ranked_chunks.append((combined_score, text))

    ranked_chunks.sort(reverse=True, key=lambda x: x[0])

    selected = ranked_chunks[:TOP_K]

    chunks = [c[1] for c in selected]
    scores = [c[0] for c in selected]

    context = "\n\n".join(chunks)

    MAX_CHARS = 1800
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS]

    if return_scores:
        return context, scores

    return context
