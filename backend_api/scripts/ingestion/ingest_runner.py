from .auth import get_access_token
from .fetch_cases import fetch_cases
from .upsert_csv import upsert_cases_to_csv


# 🔹 CONFIG (centralized, easy to change later)
CSV_OUTPUT_PATH = "data/cases_master.csv"



def run_ingestion():
    print("🚀 Starting Auto MPR ingestion pipeline")

    # 🔹 Step 1: Auth
    token = get_access_token(
        username="DUMMY",   # real creds later
        password="DUMMY"
    )

    # 🔹 Step 2: Fetch cases
    api_records = fetch_cases(token)
    print(f"📥 API records fetched: {len(api_records)}")

    # 🔹 Step 3: Upsert into CSV
    stats = upsert_cases_to_csv(
        api_records=api_records,
        csv_path=CSV_OUTPUT_PATH
    )

    print("📊 Ingestion summary:")
    print(f"   Inserted : {stats['inserted']}")
    print(f"   Updated  : {stats['updated']}")
    print(f"   Unchanged: {stats['unchanged']}")
    print("✅ Ingestion completed successfully")


if __name__ == "__main__":
    run_ingestion()
