import pandas as pd
from pathlib import Path

# -----------------------------
# Load data ONCE
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cases_training.csv"

_df = None

def load_cases_df():
    global _df
    if _df is None:
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err = None

        for enc in encodings:
            try:
                _df = pd.read_csv(DATA_PATH, encoding=enc)
                _df = _df.fillna("")
                print(f"[user_insights] Loaded CSV with encoding: {enc}")
                return _df
            except Exception as e:
                last_err = e

        raise RuntimeError(
            f"Failed to load CSV with known encodings. Last error: {last_err}"
        )

    return _df


# -----------------------------
# Utility: detect caseid vs username
# -----------------------------
def is_case_id(value: str) -> bool:
    return value.isdigit()


# -----------------------------
# Case-level details
# -----------------------------
def get_case_details(case_id: str):
    df = load_cases_df()

    case_id = int(case_id)
    case_df = df[df["caseid"] == case_id]

    if case_df.empty:
        return None

    row = case_df.iloc[0]

    return {
        "caseid": row["caseid"],
        "currentowner": row["currentowner"],
        "category": row["category"],
        "statuscode": row["statuscode"],
        "aging": row["aging"],
        "reportedon": row["reportedon"],
        "closedate": row["closedate"],
        "subject": row.get("subject", ""),
        "details": row.get("details", "")
    }


# -----------------------------
# User-level summary
# -----------------------------
def get_user_summary(owner_name: str):
    df = load_cases_df()

    # Filter by owner
    user_df = df[df["currentowner"].str.lower() == owner_name.lower()]

    if user_df.empty:
        return None

    # 🔧 CRITICAL FIX: ensure aging is numeric
    user_df = user_df.copy()
    user_df["aging"] = (
        pd.to_numeric(user_df["aging"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    total_cases = len(user_df)

    # Pending = no closed date
    pending_cases = user_df[
        (user_df["closedate"] == "") | (user_df["closedate"].isna())
    ]

    overdue_cases = user_df[user_df["aging"] > 7]
    critical_cases = user_df[user_df["aging"] > 21]

    status_counts = (
        user_df["statuscode"]
        .value_counts()
        .to_dict()
    )

    return {
        "owner": owner_name,
        "total_cases": total_cases,
        "pending_cases": len(pending_cases),
        "overdue_cases": len(overdue_cases),
        "critical_cases": len(critical_cases),
        "status_breakdown": status_counts
    }


# -----------------------------
# Unified entry point
# -----------------------------
def get_user_or_case_insights(input_value: str):
    """
    Determines whether input is a caseid or username
    and returns the appropriate data.
    """

    input_value = input_value.strip()

    if is_case_id(input_value):
        return {
            "type": "case",
            "data": get_case_details(input_value)
        }
    else:
        return {
            "type": "user",
            "data": get_user_summary(input_value)
        }
