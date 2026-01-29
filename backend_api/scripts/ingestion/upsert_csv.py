# upsert_csv.py
# ==================================================
# Snapshot → Master CSV Upsert Logic
# ==================================================
# - Schema driven (CSV_COLUMNS)
# - Primary key: caseid
# - Handles INSERT / UPDATE / UNCHANGED
# ==================================================

import csv
from pathlib import Path
from .schema import CSV_COLUMNS, PRIMARY_KEY


def upsert_cases_to_csv(snapshot_csv: str, master_csv: str):
    snapshot_csv = Path(snapshot_csv)
    master_csv = Path(master_csv)

    # -------------------------------
    # Load snapshot data
    # -------------------------------
    snapshot_rows = {}
    with open(snapshot_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            snapshot_rows[row[PRIMARY_KEY]] = row

    # -------------------------------
    # Load existing master (if exists)
    # -------------------------------
    master_rows = {}
    if master_csv.exists():
        with open(master_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                master_rows[row[PRIMARY_KEY]] = row

    inserted = 0
    updated = 0
    unchanged = 0

    # -------------------------------
    # Upsert logic
    # -------------------------------
    for caseid, snapshot_row in snapshot_rows.items():
        if caseid not in master_rows:
            master_rows[caseid] = snapshot_row
            inserted += 1
        else:
            if master_rows[caseid] != snapshot_row:
                master_rows[caseid] = snapshot_row
                updated += 1
            else:
                unchanged += 1

    # -------------------------------
    # Write updated master CSV
    # -------------------------------
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(master_rows.values())

    return {
        "inserted": inserted,
        "updated": updated,
        "unchanged": unchanged,
        "total_master_rows": len(master_rows)
    }
