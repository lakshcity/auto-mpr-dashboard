import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cases_training.csv"
INDEX_PATH = BASE_DIR / "data" / "case_index.faiss"
META_PATH  = BASE_DIR / "data" / "case_meta.pkl"

# -----------------------------
# Robust CSV Loader
# -----------------------------
def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded CSV with encoding: {enc}")
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed to load CSV. Last error: {last_err}")

# -----------------------------
# Index Builder
# -----------------------------
def build_index():
    print("=== Build Index Started ===")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

    print("Loading CSV...")
    df = load_csv_with_fallback(DATA_PATH)

    print("Detected columns:", df.columns.tolist())

    df = df.fillna("")

    # 🔧 Flexible column detection
    def find_col(names):
        for c in df.columns:
            for n in names:
                if n.lower() in c.lower():
                    return c
        return None

    case_col = find_col(["case", "id"])
    cat_col  = find_col(["category", "type"])
    sum_col  = find_col(["summary", "issue", "description"])
    res_col  = find_col(["resolution", "status", "fix", "solution"])

    # Fallback to first few columns if names are unknown
    if not case_col: case_col = df.columns[0]
    if not cat_col:  cat_col  = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    if not sum_col:  sum_col  = df.columns[2] if len(df.columns) > 2 else df.columns[0]
    if not res_col:  res_col  = df.columns[3] if len(df.columns) > 3 else df.columns[0]

    print("Using columns:")
    print("Case ID:", case_col)
    print("Category:", cat_col)
    print("Summary:", sum_col)
    print("Resolution:", res_col)

    # Combine text
    df["combined_text"] = (
        "Case ID: " + df[case_col].astype(str) + " | "
        "Category: " + df[cat_col].astype(str) + " | "
        "Issue: " + df[sum_col].astype(str) + " | "
        "Resolution: " + df[res_col].astype(str)
    )

    texts = df["combined_text"].tolist()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding cases...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
     
    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print("\n✅ Index built successfully")
    print("Saved:", INDEX_PATH)
    print("Saved:", META_PATH)

if __name__ == "__main__":
    build_index()
