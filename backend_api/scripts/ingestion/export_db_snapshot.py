import csv
import os
from datetime import date
from .fetch_from_db import fetch_cases_from_db
from .schema import CSV_COLUMNS


def export_daily_snapshot():
    records = fetch_cases_from_db()
        
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    snapshot_file = os.path.join(
        output_dir,
        f"cases_snapshot_{date.today().isoformat()}.csv"
    )

    with open(snapshot_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print(f"📄 Snapshot exported: {snapshot_file}")
    return snapshot_file


if __name__ == "__main__":
    export_daily_snapshot()
