import time
import pickle
import hashlib
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =============================
# Paths
# =============================
THIS_FILE = Path(__file__).resolve()

# project root = .../Auto-MPR-Dashboard
if "services" in str(THIS_FILE):
    BASE_DIR = THIS_FILE.parents[1]
else:
    BASE_DIR = THIS_FILE.parent

ROOT_DATA_DIR = BASE_DIR / "data"
BACKEND_DATA_DIR = BASE_DIR / "backend_api" / "data"

INDEX_PATH = ROOT_DATA_DIR / "case_index_master.faiss"
META_PATH  = ROOT_DATA_DIR / "case_meta_master.pkl"
STATE_PATH = ROOT_DATA_DIR / "case_hash_state.pkl"  # caseid -> hash

# =============================
# Config (YOU ASKED THIS)
# =============================
MAX_INDEX_ROWS = 20000       # ✅ Index only latest 20k records
DAYS_LOOKBACK = 730          # last 2 years
BATCH_SIZE = 64              # speed tuning for CPU (try 64/128)

# =============================
# Helpers
# =============================
def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[Indexer] Loaded CSV with encoding: {enc}")
            return df
        except UnicodeDecodeError:
            print(f"[Indexer] Failed encoding: {enc}")
    raise RuntimeError("Failed to load CSV with known encodings")

def get_latest_snapshot() -> Path:
    candidates = []

    # prefer parquet snapshots
    candidates += list(ROOT_DATA_DIR.glob("cases_snapshot_*.parquet"))
    candidates += list(BACKEND_DATA_DIR.glob("cases_snapshot_*.parquet"))

    # fallback to csv snapshots
    candidates += list(ROOT_DATA_DIR.glob("cases_snapshot_*.csv"))
    candidates += list(BACKEND_DATA_DIR.glob("cases_snapshot_*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No snapshots found in {ROOT_DATA_DIR} or {BACKEND_DATA_DIR}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[Indexer] 📂 Using latest snapshot: {latest}")
    return latest

def find_col(df, keywords):
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None

def _id_selector_from_ids(id_arr: np.ndarray):
    """FAISS selector helper for remove_ids()."""
    id_arr = np.asarray(id_arr, dtype="int64")
    if id_arr.size == 0:
        return None
    return faiss.IDSelectorBatch(id_arr.size, faiss.swig_ptr(id_arr))

# =============================
# Index Builder (2 years + rolling 20k + incremental)
# =============================
def build_index():
    print("\n[Indexer] 🚀 Starting Index Build (Last 2 Years + Rolling 20K + Incremental)...")
    start_time = time.time()

    file_path = get_latest_snapshot()

    # -----------------------------
    # Load snapshot
    # -----------------------------
    print("[Indexer] 📄 Loading data...")
    if file_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(file_path)
        print("[Indexer] Loaded Parquet snapshot successfully")
    else:
        df = load_csv_with_fallback(file_path)

    df = df.fillna("")
    print(f"[Indexer] Loaded Data. Shape: {df.shape}")
    print("[Indexer] Columns detected:", df.columns.tolist())

    # Identify columns
    caseid_col = find_col(df, ["caseid"])
    reportedon_col = find_col(df, ["reportedon"])
    subject_col = find_col(df, ["mpr_subject", "subject"])
    details_col = find_col(df, ["details", "description"])
    resolution_col = find_col(df, ["statuscode", "resolution", "solution"])

    if not caseid_col:
        print("[Indexer] ❌ 'caseid' column not found. Cannot proceed.")
        return

    subject_col = subject_col or df.columns[0]
    details_col = details_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    resolution_col = resolution_col or ""

    print("[Indexer] Using columns:")
    print("  CaseID    :", caseid_col)
    print("  Reported  :", reportedon_col if reportedon_col else "❌ Not found")
    print("  Subject   :", subject_col)
    print("  Details   :", details_col)
    print("  Resolution:", resolution_col if resolution_col else "❌ Not found")

    # -----------------------------
    # FILTER: last 2 years
    # -----------------------------
    if reportedon_col:
        df[reportedon_col] = pd.to_datetime(df[reportedon_col], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=DAYS_LOOKBACK)
        before = len(df)
        df = df[df[reportedon_col] >= cutoff].copy()
        after = len(df)
        print(f"[Indexer] 🗓️ Last-{DAYS_LOOKBACK}days filter: {before} -> {after}")
    else:
        print("[Indexer] ⚠️ reportedon not found; skipping 2-year filter.")

    # Drop rows with empty text
    before = len(df)
    df = df[(df[subject_col].astype(str).str.strip() != "") | (df[details_col].astype(str).str.strip() != "")].copy()
    after = len(df)
    print(f"[Indexer] 🧹 Dropped empty text rows: {before} -> {after}")

    # -----------------------------
    # CAP: latest 20k records (rolling window)
    # -----------------------------
    before = len(df)
    if reportedon_col and reportedon_col in df.columns:
        df = df.sort_values(by=reportedon_col, ascending=False).head(MAX_INDEX_ROWS).copy()
    else:
        df = df.head(MAX_INDEX_ROWS).copy()
    after = len(df)
    print(f"[Indexer] 🎯 Rolling cap to latest {MAX_INDEX_ROWS}: {before} -> {after}")

    # -----------------------------
    # Build combined_text + hash
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

    df["caseid_str"] = df[caseid_col].astype(str)
    df["text_hash"] = df["combined_text"].astype(str).apply(md5_text)

    # -----------------------------
    # Load previous state if exists
    # -----------------------------
    prev_state = {}
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "rb") as f:
                prev_state = pickle.load(f)
        except Exception:
            prev_state = {}

    first_build = (not INDEX_PATH.exists()) or (not META_PATH.exists()) or (not STATE_PATH.exists())

    # Determine deltas within the rolling 20k window
    is_new = ~df["caseid_str"].isin(prev_state.keys())
    is_mod = df["caseid_str"].isin(prev_state.keys()) & (df["text_hash"] != df["caseid_str"].map(prev_state))

    new_df = df[is_new].copy()
    mod_df = df[is_mod].copy()

    print(f"[Indexer] ➕ New cases (in window): {len(new_df)}")
    print(f"[Indexer] ♻️ Modified cases (in window): {len(mod_df)}")

    # Also compute "evictions": ids that were previously indexed but are NOT in the latest 20k window now
    current_ids_set = set(df["caseid_str"].tolist())
    prev_ids_set = set(prev_state.keys())
    evict_ids = sorted(list(prev_ids_set - current_ids_set))
    print(f"[Indexer] 🧯 Evicted (out of latest {MAX_INDEX_ROWS}): {len(evict_ids)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # =============================
    # FIRST BUILD (only rolling 20k)
    # =============================
    if first_build:
        print(f"[Indexer] 🧱 First build detected. Building full index for latest {MAX_INDEX_ROWS}...")
        work_df = df.copy()

        texts = work_df["combined_text"].tolist()
        print(f"[Indexer] 🧠 Encoding {len(texts)} cases...")
        emb = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE
        ).astype("float32")

        dim = emb.shape[1]
        base = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap2(base)

        ids = pd.to_numeric(work_df["caseid_str"], errors="coerce").fillna(0).astype("int64").values
        index.add_with_ids(emb, ids)

        ROOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))

        with open(META_PATH, "wb") as f:
            pickle.dump(work_df.to_dict(orient="records"), f)

        state = dict(zip(work_df["caseid_str"].tolist(), work_df["text_hash"].tolist()))
        with open(STATE_PATH, "wb") as f:
            pickle.dump(state, f)

        elapsed = round(time.time() - start_time, 2)
        print(f"\n[Indexer] ✅ Full index built successfully (latest {MAX_INDEX_ROWS})")
        print("[Indexer] Saved:", INDEX_PATH)
        print("[Indexer] Saved:", META_PATH)
        print("[Indexer] Saved:", STATE_PATH)
        print(f"[Indexer] ⏱️ Time taken: {elapsed}s")
        return

    # =============================
    # INCREMENTAL UPDATE
    # =============================
    print("[Indexer] 🔁 Loading existing index + metadata...")
    index = faiss.read_index(str(INDEX_PATH))

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    meta_df = pd.DataFrame(meta)

    # Ensure meta has caseid_str
    if "caseid_str" not in meta_df.columns:
        if caseid_col in meta_df.columns:
            meta_df["caseid_str"] = meta_df[caseid_col].astype(str)
        elif "caseid" in meta_df.columns:
            meta_df["caseid_str"] = meta_df["caseid"].astype(str)
        else:
            meta_df["caseid_str"] = ""

    # 1) Remove evicted IDs (rolling window maintenance)
    if len(evict_ids) > 0:
        evict_arr = pd.to_numeric(pd.Series(evict_ids), errors="coerce").fillna(0).astype("int64").values
        sel = _id_selector_from_ids(evict_arr)
        if sel is not None:
            index.remove_ids(sel)
        meta_df = meta_df[~meta_df["caseid_str"].isin(evict_ids)].copy()

    # 2) Remove modified IDs first (so they can be re-added)
    if len(mod_df) > 0:
        mod_ids = pd.to_numeric(mod_df["caseid_str"], errors="coerce").fillna(0).astype("int64").values
        sel = _id_selector_from_ids(mod_ids)
        if sel is not None:
            index.remove_ids(sel)
        meta_df = meta_df[~meta_df["caseid_str"].isin(mod_df["caseid_str"].tolist())].copy()

    # 3) Add new + modified
    delta_df = pd.concat([new_df, mod_df], ignore_index=True)

    if len(delta_df) > 0:
        texts = delta_df["combined_text"].tolist()
        print(f"[Indexer] 🧠 Encoding delta {len(texts)} cases...")
        emb = model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=BATCH_SIZE
        ).astype("float32")

        ids = pd.to_numeric(delta_df["caseid_str"], errors="coerce").fillna(0).astype("int64").values
        index.add_with_ids(emb, ids)

        meta_df = pd.concat([meta_df, delta_df], ignore_index=True)

    # Save updated artifacts
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_df.to_dict(orient="records"), f)

    # Update state (keep only current window)
    new_state = {}
    for _, r in df.iterrows():
        new_state[str(r["caseid_str"])] = str(r["text_hash"])

    with open(STATE_PATH, "wb") as f:
        pickle.dump(new_state, f)

    elapsed = round(time.time() - start_time, 2)
    print("\n[Indexer] ✅ Incremental update complete (rolling window maintained)")
    print("[Indexer] Saved:", INDEX_PATH)
    print("[Indexer] Saved:", META_PATH)
    print("[Indexer] Saved:", STATE_PATH)
    print(f"[Indexer] ⏱️ Time taken: {elapsed}s")

# =============================
# Entry Point
# =============================
if __name__ == "__main__":
    build_index()