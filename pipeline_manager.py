import logging
import time
import sys
import os
import subprocess  # <--- NEW: Allows running Git commands

# ... (Keep your existing path setup and imports) ...
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
log_file_path = os.path.join(current_dir, "pipeline.log")

from backend_api.scripts.ingestion import export_db_snapshot
from backend_api.scripts.ingestion import run_upsert_snapshot

try:
    import indexer
except ImportError:
    try:
        from services import indexer
    except ImportError:
        print("❌ CRITICAL ERROR: Could not find 'indexer.py'.")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)

# ==========================================
# NEW FUNCTION: Auto-Git Pusher
# ==========================================
def push_to_github():
    try:
        logging.info("4. Pushing updates to GitHub...")
        print("🚀 Pushing new data to GitHub...")

        # 1. Add changes
        subprocess.run(["git", "add", "data/*.csv", "data/*.faiss", "data/*.pkl"], check=True)
        
        # 2. Commit (Ignore error if no changes exist)
        subprocess.run(["git", "commit", "-m", "🤖 Auto-Update: Fresh Data & Index"], check=False)
        
        # 3. Push to master (Or your specific branch)
        subprocess.run(["git", "push", "origin", "master"], check=True)
        
        logging.info("✅ Successfully pushed to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Git Push Failed: {e}")
        return False

def run_full_pipeline():
    logging.info("=== STARTING AUTOMATED PIPELINE ===")
    
    try:
        # Step 1: Fetch
        logging.info("1. Fetching Data from DB...")
        export_db_snapshot.export_daily_snapshot()
        
        # Step 2: Upsert
        logging.info("2. Merging Snapshot to Master CSV...")
        run_upsert_snapshot.run_snapshot_upsert()
        
        # Step 3: Index
        logging.info("3. Rebuilding FAISS Index...")
        indexer.build_index()

        # Step 4: Git Push (The Magic Step)
        push_to_github()  # <--- CALLING THE NEW FUNCTION
        
        logging.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        logging.error(f"!!! PIPELINE FAILED: {str(e)}")
        print(f"❌ Pipeline Failed: {e}")
        return False

if __name__ == "__main__":
    run_full_pipeline()