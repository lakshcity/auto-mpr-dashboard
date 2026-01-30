import time
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# =============================
# Paths
# =============================
BASE_DIR = Path(__file__).resolve().parents[1]

SCALE = "master"  # FORCE master

DATA_PATH = BASE_DIR / "backend_api" / "data" / "cases_master.csv"
INDEX_PATH = BASE_DIR / "data" / "case_index_master.faiss"
META_PATH = BASE_DIR / "data" / "case_meta_master.pkl"
# =============================
# Robust CSV Loader
# =============================
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"Failed with encoding: {enc}")

    raise RuntimeError("Failed to load CSV with known encodings")

# =============================
# Index Builder
# =============================
def build_index():
    print("=== Build Index Started ===")
    start_time = time.time()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    print("Loading CSV...")
    df = load_csv_with_fallback(DATA_PATH)

    print("Detected columns:", df.columns.tolist())
    df = df.fillna("")

    # -----------------------------
    # Flexible column detection
    # -----------------------------
    def find_col(keywords):
        for col in df.columns:
            for kw in keywords:
                if kw.lower() in col.lower():
                    return col
        return None

    case_col = find_col(["case", "id"]) or df.columns[0]
    cat_col  = find_col(["category", "type"]) or df.columns[1]
    sum_col  = find_col(["summary", "issue", "description", "details", "subject"]) or df.columns[2]
    res_col  = find_col(["resolution", "status", "fix", "solution"]) or df.columns[3]

    print("Using columns:")
    print("Case ID:", case_col)
    print("Category:", cat_col)
    print("Summary:", sum_col)
    print("Resolution:", res_col)

    # -----------------------------
    # Combine text for embeddings
    # -----------------------------
    df["combined_text"] = (
        "Case ID: " + df[case_col].astype(str) + " | "
        "Category: " + df[cat_col].astype(str) + " | "
        "Issue: " + df[sum_col].astype(str) + " | "
        "Resolution: " + df[res_col].astype(str)
    )

    texts = df["combined_text"].tolist()

    # -----------------------------
    # Embedding
    # -----------------------------
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding cases...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype("float32")

    # -----------------------------
    # FAISS Index
    # -----------------------------
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    # -----------------------------
    # Timing End
    # -----------------------------
    end_time = time.time()

    print("\n✅ Index built successfully")
    print("Saved:", INDEX_PATH)
    print("Saved:", META_PATH)
    print(f"\n⏱️ Index build time (25k records): {round(end_time - start_time, 2)} seconds")


# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    build_index()
