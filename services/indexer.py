import time
import pickle
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# =============================
# Paths (robust, no assumptions)
# =============================
THIS_FILE = Path(__file__).resolve()

# Handle running from repo root OR inside backend_api/services
if "backend_api" in str(THIS_FILE):
    BASE_DIR = THIS_FILE.parents[2]
else:
    BASE_DIR = THIS_FILE.parents[1]

DATA_PATH = BASE_DIR / "backend_api" / "data" / "cases_master.csv"
INDEX_PATH = BASE_DIR / "data" / "case_index_master.faiss"
META_PATH  = BASE_DIR / "data" / "case_meta_master.pkl"

# =============================
# Robust CSV Loader
# =============================
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[Indexer] Loaded CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"[Indexer] Failed encoding: {enc}")

    raise RuntimeError("Failed to load CSV with known encodings")

# =============================
# Index Builder
# =============================
def build_index():
    print("\n[Indexer] 🚀 Starting Master Index Rebuild...")
    start_time = time.time()

    if not DATA_PATH.exists():
        print(f"[Indexer] ❌ CSV not found at: {DATA_PATH}")
        return

    # -----------------------------
    # Load CSV
    # -----------------------------
    print("[Indexer] 📄 Loading CSV...")
    df = load_csv_with_fallback(DATA_PATH)
    df = df.fillna("")

    print("[Indexer] Columns detected:", df.columns.tolist())

    # -----------------------------
    # Flexible column detection
    # -----------------------------
    def find_col(keywords):
        for col in df.columns:
            for kw in keywords:
                if kw.lower() in col.lower():
                    return col
        return None

    subject_col = find_col(["subject", "summary", "issue", "description", "details"])
    details_col = find_col(["details", "description", "content"])
    resolution_col = find_col(["resolution", "solution", "fix", "answer", "status"])

    # Safe fallbacks
    subject_col = subject_col or df.columns[0]
    details_col = details_col or df.columns[1]
    resolution_col = resolution_col or ""

    print("[Indexer] Using columns:")
    print("  Subject   :", subject_col)
    print("  Details   :", details_col)
    print("  Resolution:", resolution_col if resolution_col else "❌ Not found")

    # -----------------------------
    # Combine text for embeddings
    # -----------------------------
    print("[Indexer] 🧩 Building combined text...")

    if resolution_col:
        df["combined_text"] = (
            "Subject: " + df[subject_col].astype(str) + " | "
            "Details: " + df[details_col].astype(str) + " | "
            "Resolution: " + df[resolution_col].astype(str)
        )
    else:
        df["combined_text"] = (
            "Subject: " + df[subject_col].astype(str) + " | "
            "Details: " + df[details_col].astype(str)
        )

    texts = df["combined_text"].tolist()

    # -----------------------------
    # Embedding
    # -----------------------------
    print(f"[Indexer] 🧠 Encoding {len(texts)} cases...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # -----------------------------
    # FAISS Index
    # -----------------------------
    print("[Indexer] 🏗️ Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    elapsed = round(time.time() - start_time, 2)

    print("\n[Indexer] ✅ Index built successfully")
    print("[Indexer] Saved:", INDEX_PATH)
    print("[Indexer] Saved:", META_PATH)
    print(f"[Indexer] ⏱️ Time taken: {elapsed}s")

# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    build_index()
