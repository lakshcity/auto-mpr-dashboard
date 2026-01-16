import pandas as pd
from pathlib import Path

# -----------------------------
# Helpers
# -----------------------------
def _to_int(val):
    try:
        return int(val)
    except Exception:
        return 0


# -----------------------------
# Load data ONCE (cached)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cases_training.csv"

_df = None


def load_cases_df():
    global _df
    if _df is not None:
        return _df

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

    try:
        case_id = int(case_id)
    except Exception:
        return None

    case_df = df[df["caseid"] == case_id]

    if case_df.empty:
        return None

    row = case_df.iloc[0]

    return {
        "caseid": row["caseid"],
        "currentowner": row["currentowner"],
        "category": row["category"],
        "statuscode": row["statuscode"],
        "aging": _to_int(row["aging"]),
        "reportedon": row["reportedon"],
        "closedate": row["closedate"],
        "subject": row.get("subject", ""),
        "details": row.get("details", ""),
    }


# -----------------------------
# User-level summary
# -----------------------------
def get_user_summary(owner_name: str):
    df = load_cases_df()

    user_df = df[
        df["currentowner"].str.lower() == owner_name.lower()
    ].copy()

    if user_df.empty:
        return None

    # Ensure numeric aging
    user_df["aging_num"] = user_df["aging"].apply(_to_int)

    total_cases = len(user_df)

    pending_cases = user_df[
        (user_df["closedate"] == "") | (user_df["closedate"].isna())
    ]

    overdue_cases = user_df[user_df["aging_num"] > 7]
    critical_cases = user_df[user_df["aging_num"] > 21]

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
        "status_breakdown": status_counts,
    }


# -----------------------------
# Internal helper for case lists
# -----------------------------
def _get_user_cases(owner_name: str):
    df = load_cases_df()

    user_df = df[
        df["currentowner"].str.lower() == owner_name.lower()
    ].copy()

    if user_df.empty:
        return pd.DataFrame()

    user_df["aging_num"] = user_df["aging"].apply(_to_int)
    return user_df


# -----------------------------
# Focused case buckets (ALWAYS return list)
# -----------------------------
def get_pending_cases(owner_name: str, top_n: int = 3):
    user_df = _get_user_cases(owner_name)

    if user_df.empty:
        return []

    pending_df = user_df[
        (user_df["closedate"] == "") | (user_df["closedate"].isna())
    ]

    if pending_df.empty:
        return []

    pending_df = pending_df.sort_values(
        by="aging_num",
        ascending=False
    )

    return pending_df.head(top_n).to_dict(orient="records")


def get_overdue_cases(owner_name: str, top_n: int = 3):
    user_df = _get_user_cases(owner_name)

    if user_df.empty:
        return []

    overdue_df = user_df[user_df["aging_num"] > 7]

    if overdue_df.empty:
        return []

    overdue_df = overdue_df.sort_values(
        by="aging_num",
        ascending=False
    )

    return overdue_df.head(top_n).to_dict(orient="records")


def get_critical_cases(owner_name: str, top_n: int = 3):
    user_df = _get_user_cases(owner_name)

    if user_df.empty:
        return []

    critical_df = user_df[user_df["aging_num"] > 21]

    if critical_df.empty:
        return []

    critical_df = critical_df.sort_values(
        by="aging_num",
        ascending=False
    )

    return critical_df.head(top_n).to_dict(orient="records")


# -----------------------------
# Unified entry point
# -----------------------------
def get_user_or_case_insights(input_value: str):
    input_value = input_value.strip()

    if is_case_id(input_value):
        return {
            "type": "case",
            "data": get_case_details(input_value),
        }
    else:
        return {
            "type": "user",
            "data": get_user_summary(input_value),
        }
