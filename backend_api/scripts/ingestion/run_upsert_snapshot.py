# run_upsert_snapshot.py
# ==================================================
# Orchestrates snapshot → master upsert
# ==================================================

from pathlib import Path
from .upsert_csv import upsert_cases_to_csv

DATA_DIR = Path("data")


def run_snapshot_upsert():
    # Find latest snapshot
    snapshots = sorted(DATA_DIR.glob("cases_snapshot_*.csv"))
    if not snapshots:
        raise RuntimeError("❌ No snapshot CSV found in data/ directory")

    latest_snapshot = snapshots[-1]
    master_csv = DATA_DIR / "cases_master.csv"

    print(f"📥 Using snapshot : {latest_snapshot.name}")
    print(f"📊 Master target : {master_csv.name}")

    summary = upsert_cases_to_csv(
        snapshot_csv=str(latest_snapshot),
        master_csv=str(master_csv)
    )

    print("📊 Upsert summary:")
    print(f"   Inserted   : {summary['inserted']}")
    print(f"   Updated    : {summary['updated']}")
    print(f"   Unchanged  : {summary['unchanged']}")
    print(f"   Total rows : {summary['total_master_rows']}")


if __name__ == "__main__":
    run_snapshot_upsert()
