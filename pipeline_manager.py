import logging
import sys
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# =========================
# Setup paths
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(current_dir).resolve()
DATA_DIR = ROOT_DIR / "data"

# Ensure imports resolve from project root
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

log_file_path = os.path.join(current_dir, "pipeline.log")

# =========================
# Imports
# =========================
from backend_api.scripts.ingestion import export_db_snapshot

try:
    import indexer
except ImportError:
    from services import indexer

# =========================
# Logging (UTF-8 safe on Windows)
# =========================
# This prevents UnicodeEncodeError when logging emojis on Windows console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

# =========================
# Cleanup old snapshots
# =========================
def cleanup_old_snapshots(days_to_keep=2):
    """
    Deletes snapshot files older than X days.
    Supports both CSV and Parquet snapshots.
    Snapshot naming expected:
      cases_snapshot_YYYY-MM-DD.csv
      cases_snapshot_YYYY-MM-DD.parquet
    """
    try:
        logging.info(f"🧹 Scanning for old snapshots in: {DATA_DIR}")
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        count = 0

        # Support both parquet and csv (your earlier pipeline produced CSV snapshots)
        files = list(DATA_DIR.glob("cases_snapshot_*.parquet")) + list(DATA_DIR.glob("cases_snapshot_*.csv"))

        for file_path in files:
            try:
                # Extract date from filename: cases_snapshot_2026-02-06.csv / .parquet
                date_str = file_path.stem.replace("cases_snapshot_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if file_date < cutoff_date:
                    os.remove(file_path)
                    logging.info(f"🗑️ Deleted: {file_path.name}")
                    count += 1
            except Exception:
                continue

        logging.info(f"✅ Cleanup finished. {count} files removed.")
        return True

    except Exception as e:
        logging.error(f"❌ Cleanup failed: {e}")
        return False

# =========================
# Git Push (safe: no snapshots)
# =========================
def push_to_github():
    """
    Push ONLY the files needed for Streamlit Cloud:
      - data/case_index_master.faiss
      - data/case_meta_master.pkl
      - (optional) data/case_hash_state.pkl (if you use incremental indexing later)
    This avoids GitHub 100MB limit triggered by large snapshots.
    """
    try:
        logging.info("🚀 Syncing to GitHub...")

        # Add only safe artifacts (NO snapshots)
        files_to_add = [
            "data/case_index_master.faiss",
            "data/case_meta_master.pkl",
            "data/case_hash_state.pkl",   # optional, harmless if not present
        ]

        # Stage only existing files
        for fp in files_to_add:
            full_path = ROOT_DIR / fp
            if full_path.exists():
                subprocess.run(["git", "add", fp], check=False)

        # Commit (won't fail pipeline if no changes)
        commit_msg = f"🤖 Auto-Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=False)

        # Push master; fallback main
        try:
            subprocess.run(["git", "push", "origin", "master"], check=True)
        except Exception:
            subprocess.run(["git", "push", "origin", "main"], check=True)

        logging.info("✅ GitHub Sync Complete.")
        return True

    except Exception as e:
        logging.error(f"❌ Git Push Failed: {e}")
        return False

# =========================
# Pipeline Runner
# =========================
def run_full_pipeline():
    logging.info("=== STARTING PIPELINE ===")
    try:
        # 1) Fetch snapshot from DB (CSV/Parquet depending on your exporter)
        ok = export_db_snapshot.export_daily_snapshot()
        if ok is False:
            logging.error("❌ DB Fetch failed.")
            return

        # 2) Index (your indexer auto-picks latest snapshot now)
        indexer.build_index()

        # 3) Cleanup old snapshots
        cleanup_old_snapshots(days_to_keep=2)

        # 4) Push artifacts to GitHub (NO snapshots)
        push_to_github()

        logging.info("=== SUCCESS ===")

    except Exception as e:
        logging.error(f"❌ Crash: {e}")

if __name__ == "__main__":
    run_full_pipeline()