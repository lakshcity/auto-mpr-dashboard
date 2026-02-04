

import requests
import csv
import sys
import os
import datetime


REDASH_URL = "https://redash.businessnext.com"
API_KEY = "dBGaJVGwbgkKzt0hukVEWnelw0vdCme746veXuNJ"
QUERY_ID = 305

DATA_DIR = "/home/user/Auto-MPR-Dashboard/data"
OUTPUT_FILE = os.path.join(DATA_DIR, "redash_latest.csv")
LOG_FILE = os.path.join(DATA_DIR, "redash_cron.log")


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} | {msg}\n")

# =========================
# ENSURE DIRECTORY
# =========================
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# FETCH DATA
# =========================
url = f"{REDASH_URL}/api/queries/{QUERY_ID}/results.json"

headers = {
    "Authorization": f"Key {API_KEY}",
    "Content-Type": "application/json"
}

try:
    response = requests.get(url, headers=headers, timeout=30)
except Exception as e:
    log(f"Request failed: {e}")
    sys.exit(1)

if response.status_code != 200:
    log(f"Failed to fetch data | Status {response.status_code} | {response.text}")
    sys.exit(1)

data = response.json()

rows = data.get("query_result", {}).get("data", {}).get("rows", [])

if not rows:
    log("No data returned from query")
    sys.exit(0)

log(f"Rows fetched: {len(rows)}")

# =========================
# DELETE OLD FILE
# =========================
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

# =========================
# WRITE CSV (overwrite)
# =========================
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

log(f"CSV updated successfully: {OUTPUT_FILE}")

print("✅ Redash CSV refreshed successfully")
