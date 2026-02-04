import logging
import time
import sys
import os
from pathlib import Path

# ==========================================
# FIX 1: Add Root Directory to Python Path
# ==========================================
# Get the absolute path of the folder this script is in
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# ==========================================
# FIX 2: Absolute Path for Log File
# ==========================================
# This forces the log file to be created RIGHT HERE, next to the script
log_file_path = os.path.join(current_dir, "pipeline.log")

# ==========================================
# Imports
# ==========================================
from backend_api.scripts.ingestion import export_db_snapshot
from backend_api.scripts.ingestion import run_upsert_snapshot

# Try importing indexer from Root (Standard) or Services (Alternate)
try:
    import indexer
except ImportError:
    try:
        # Fallback if you moved indexer.py to services/ folder
        from services import indexer
    except ImportError:
        print("❌ CRITICAL ERROR: Could not find 'indexer.py'.")
        print("   Please ensure 'indexer.py' is in the SAME folder as this script.")
        sys.exit(1)

# Setup Logging
# We now use the absolute 'log_file_path' we defined above
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # <--- FORCE PATH HERE
        logging.StreamHandler()
    ]
)

def run_full_pipeline():
    print(f"📝 Logging to: {log_file_path}") # Tell user exactly where the log is
    logging.info("=== STARTING AUTOMATED PIPELINE ===")
    
    try:
        # Step 1: Fetch from DB -> Create Daily Snapshot
        logging.info("1. Fetching Data from DB...")
        export_db_snapshot.export_daily_snapshot()
        
        # Step 2: Merge Snapshot -> Master CSV (Upsert)
        logging.info("2. Merging Snapshot to Master CSV...")
        run_upsert_snapshot.run_snapshot_upsert()
        
        # Step 3: Rebuild Vector Index
        logging.info("3. Rebuilding FAISS Index...")
        indexer.build_index()
        
        logging.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        logging.error(f"!!! PIPELINE FAILED: {str(e)}")
        # Print to console as well so you see it immediately
        print(f"❌ Pipeline Failed: {e}")
        return False

if __name__ == "__main__":
    run_full_pipeline()