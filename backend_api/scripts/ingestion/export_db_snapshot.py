# export_db_snapshot.py
# ==================================================
# Daily DB → CSV Snapshot Export
# ==================================================

import csv
from datetime import date
from pathlib import Path

from .fetch_from_db import fetch_cases_from_db
from .schema import CSV_COLUMNS

# Base data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def export_daily_snapshot():
    """
    Fetch cases from DataMart and export
    a single daily snapshot CSV.
    """

    today = date.today().isoformat()
    snapshot_file = DATA_DIR / f"cases_snapshot_{today}.csv"

    # Fetch data from DB
    records = fetch_cases_from_db()

    if not records:
        print("⚠️ No records fetched from DB")
        return

    # Write CSV snapshot
    with open(snapshot_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"  # ignore unexpected DB fields safely
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"📄 Snapshot exported successfully: {snapshot_file}")
    print(f"📊 Records written: {len(records)}")


if __name__ == "__main__":
    export_daily_snapshot()
