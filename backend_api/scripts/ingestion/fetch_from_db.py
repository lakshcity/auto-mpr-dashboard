# fetch_from_db.py

import psycopg2
from psycopg2.extras import RealDictCursor
from .db_config import DB_CONFIG, CASES_QUERY


def fetch_cases_from_db():
    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(CASES_QUERY)
        rows = cursor.fetchall()

        print(f"✅ Fetched {len(rows)} records from Data Mart")
        return rows

    except Exception as e:
        print("❌ DB fetch failed")
        print(e)
        return []

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
