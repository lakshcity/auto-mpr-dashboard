import csv
import os
from typing import List, Dict

from .schema import CSV_COLUMNS, API_TO_CSV_FIELD_MAP


PRIMARY_KEY = "caseid"


def normalize_value(value):
    """Normalize values for CSV comparison."""
    if value is None:
        return ""
    return str(value).strip()


def map_api_record_to_csv(api_record: Dict) -> Dict:
    """
    Convert a raw API record into a CSV-aligned dict
    using the canonical schema.
    """
    csv_row = {}

    for api_field, csv_field in API_TO_CSV_FIELD_MAP.items():
        csv_row[csv_field] = normalize_value(api_record.get(api_field))

    return csv_row


def upsert_cases_to_csv(
    api_records: List[Dict],
    csv_path: str
) -> Dict[str, int]:
    """
    Upsert API records into CSV.

    Returns stats:
    {
        "inserted": int,
        "updated": int,
        "unchanged": int
    }
    """

    existing_rows = {}
    stats = {"inserted": 0, "updated": 0, "unchanged": 0}

    # 🔹 Load existing CSV (if present)
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row[PRIMARY_KEY]] = row

    # 🔹 Process incoming API records
    for record in api_records:
        csv_row = map_api_record_to_csv(record)
        case_id = csv_row[PRIMARY_KEY]

        if case_id not in existing_rows:
            existing_rows[case_id] = csv_row
            stats["inserted"] += 1
        else:
            existing_row = existing_rows[case_id]

            if existing_row != csv_row:
                existing_rows[case_id] = csv_row
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1

    # 🔹 Write CSV atomically
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    temp_path = csv_path + ".tmp"

    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows.values())

    os.replace(temp_path, csv_path)

    return stats
