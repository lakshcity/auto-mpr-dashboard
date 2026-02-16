# =====================================================
# 🚀 RETRIEVER: PDF RAG + REDASH HYBRID (OPTIMIZED)
# =====================================================

import pickle
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from core.config import PDF_INDEX, PDF_META, EMBED_MODEL, TOP_K
from services.feedback_manager import get_subject_success_rate

# =====================================================
# 🔹 1️⃣ LOAD PDF RAG RESOURCES (FOR RECOMMENDATION)
# =====================================================

print("🔄 Loading embedding model...")
MODEL = SentenceTransformer(EMBED_MODEL)

print("🔄 Loading PDF FAISS index...")
INDEX = faiss.read_index(str(PDF_INDEX))

with open(PDF_META, "rb") as f:
    METADATA = pickle.load(f)

print("✅ PDF Retriever Ready")


# =====================================================
# 🔹 2️⃣ LOAD REDASH FILE (MULTI-FORMAT SUPPORT)
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

REDASH_DF = pd.DataFrame()
REDASH_INDEX = None
REDASH_READY = False


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

                # Normalize columns
                REDASH_DF.columns = (
                    REDASH_DF.columns
                    .str.strip()
                    .str.lower()
                )

                print(f"✅ Loaded Redash file: {file_path.name}")
                print("Columns:", REDASH_DF.columns.tolist())
                return
            except Exception as e:
                print(f"❌ Error loading {file_path.name}: {e}")

    print("❌ No Redash file found.")


# Load Redash
load_redash_file()




# =====================================================
# 🔹 4️⃣ HYBRID SEARCH (KEYWORD + SEMANTIC)
# =====================================================

def search_redash_mpr(query, top_k=5):

    global REDASH_INDEX

    if REDASH_DF.empty or not query.strip():
        return []

    subject_col = "mpr_subject"

    if subject_col not in REDASH_DF.columns:
        return []

    query_lower = query.lower()

    # ---------------------------------------
    # 1️⃣ KEYWORD MATCH FIRST (FAST)
    # ---------------------------------------
    keyword_df = REDASH_DF[
        REDASH_DF[subject_col]
        .astype(str)
        .str.lower()
        .str.contains(query_lower, na=False)
    ]

    if not keyword_df.empty:
        results = keyword_df.head(top_k).to_dict("records")

        enhanced_results = []

        for r in results:
            r["confidence"] = 95

            subject = r.get("mpr_subject", "")
            success_rate = get_subject_success_rate(subject)
            reward_scaled = success_rate * 20

            adaptive_score = (
                0.7 * r["confidence"] +
                0.3 * reward_scaled
            )

            if success_rate < 1.5:
                adaptive_score*= 0.8

            composite_confidence = (
                0.6 * r["confidence"] +
                0.4 * reward_scaled
            )

            r["adaptive_score"] = round(adaptive_score, 2)
            r["composite_confidence"] = round(composite_confidence, 2)

            enhanced_results.append(r)

        return enhanced_results

    # ---------------------------------------
    # 2️⃣ SEMANTIC MATCH (LAZY BUILD)
    # ---------------------------------------

    if REDASH_INDEX is None:
        print("⚙️ Building Redash semantic index (lazy build)...")

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

        print("✅ Redash semantic index ready")

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

        row["confidence"] = round(confidence, 2)

        # ---------------------------------------
        # 🔥 Adaptive Learning Layer
        # ---------------------------------------
        subject = row.get("mpr_subject", "")
        success_rate = get_subject_success_rate(subject)

        # Scale reward (0–5) to 0–100 range
        reward_scaled = success_rate * 20

        final_score = (
            0.7 * row["confidence"] +
            0.3 * reward_scaled
        )

        # ---------------------------------------
        # 🔥 Penalty Memory Layer
        # ---------------------------------------
        if success_rate < 1.5:
            final_score *= 0.8   # reduce score by 20%

        row["adaptive_score"] = round(final_score, 2)


        # Composite confidence for UI display
        composite_confidence = (
            0.6 * row["confidence"] +
            0.4 * reward_scaled
        )

        row["composite_confidence"] = round(composite_confidence, 2)


        results.append(row)


    return results



# =====================================================
# 🔹 5️⃣ PDF CONTEXT RETRIEVAL (FOR RAG)
# =====================================================

def retrieve_context(query, return_scores=False):

    if not query.strip():
        return "", []

    query_vec = MODEL.encode([query]).astype("float32")
    distances, indices = INDEX.search(query_vec, TOP_K)

    chunks = []
    scores = []

    for rank, idx in enumerate(indices[0]):
        if 0 <= idx < len(METADATA):
            text = METADATA[idx].get("text", "")
            if text:
                chunks.append(text)
                # Convert L2 distance to similarity score
                score = 1 / (1 + distances[0][rank])
                scores.append(score)

    # Compress context size
    context = "\n\n".join(chunks)
    MAX_CHARS = 1500
    if len(context) > MAX_CHARS:
        context = context[:MAX_CHARS]

    if return_scores:
        return context, scores

    return context


