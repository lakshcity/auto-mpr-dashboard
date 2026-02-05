import pandas as pd
from pathlib import Path
import os

# =========================
# Hardcoded Owner Mapping
# =========================
OWNER_MAP = {
    "aditya singh": 781,
    "akarsh bhatt": 6039,
    "amit anand": 827,
    "anubhav gupta": 4310,
    "deepali kumari": 5249,
    "dheeraj": 4776,
    "himanshu padaliya": 4019,
    "laksh gupta": 6035,
    "lisha gupta": 5443,
    "niharika verma": 4185,
    "sagar verma": 4777,
    "testinguserkam@6": 5437,
    "veeresh kumar verma": 5926,
    "vikas ojha": 3898,
    "vishal kumar": 5736,
    "vivek kumar": 3701,
    "yajurva tiwari": 3520
}

# -----------------------------
# Configuration: Business Logic
# -----------------------------
TERMINAL_STATUSES = ["Resolved", "Closed", "Invalid"]

# -----------------------------
# Helpers
# -----------------------------
def _to_int(val):
    try:
        if isinstance(val, str):
            clean_val = val.lower().replace("d", "").strip()
            return int(float(clean_val))
        return int(float(val))
    except Exception:
        return 0

def _to_float(val):
    try:
        if isinstance(val, str):
            clean_val = val.lower().replace("d", "").strip()
            return float(clean_val)
        return float(val)
    except Exception:
        return 0.0

def _raw_age_to_days(raw_val):
    """
    Fallback converter for 'ageing' column when dates are missing.
    NOTE: In your CSV, many ageing values behave like minutes (e.g. 18707),
    so if the number is large, we treat it as minutes and convert to days.
    """
    x = _to_float(raw_val)
    if x <= 0:
        return 0.0

    # Heuristic: huge values are almost always minutes in your dataset
    # Example: 18707 minutes ~ 13 days (matches reportedon->closeddate range)
    if x > 1000:
        return x / 1440.0

    # Otherwise treat as days
    return x

# -----------------------------
# Load data ONCE (cached)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

POSSIBLE_PATHS = [
    BASE_DIR / "backend_api" / "data" / "cases_master.csv",
    BASE_DIR / "data" / "cases_master.csv",
    BASE_DIR / "app" / "data" / "cases_master.csv",
    BASE_DIR / "cases_master.csv"
]

_df = None

def get_actual_data_path():
    for path in POSSIBLE_PATHS:
        if path.exists():
            print(f"[user_insights] Found data at: {path}")
            return path
    raise FileNotFoundError(
        f"Could not find 'cases_master.csv'. Checked: {[str(p) for p in POSSIBLE_PATHS]}"
    )

def load_cases_df():
    global _df
    if _df is not None:
        return _df

    real_data_path = get_actual_data_path()
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    for enc in encodings:
        try:
            temp_df = pd.read_csv(real_data_path, encoding=enc)

            # NOTE: DO NOT rename MPR_Subject -> subject directly because CSV already has 'subject'
            # This caused: "DataFrame columns are not unique" warnings. (seen in Streamlit logs)
            column_mapping = {
                "ownername": "currentowner",
                "ageing": "ageing",          # keep original name
                "Statuscode": "Statuscode"   # keep raw column if present
            }
            temp_df.rename(columns=column_mapping, inplace=True)

            # ---- FIX DUPLICATE SUBJECT COLUMN ----
            # If both 'subject' and 'MPR_Subject' exist, keep 'subject' and rename MPR_Subject to 'mpr_subject'
            if "subject" in temp_df.columns and "MPR_Subject" in temp_df.columns:
                temp_df.rename(columns={"MPR_Subject": "mpr_subject"}, inplace=True)
            elif "MPR_Subject" in temp_df.columns and "subject" not in temp_df.columns:
                temp_df.rename(columns={"MPR_Subject": "subject"}, inplace=True)

            # Ensure required columns exist
            if "currentowner" not in temp_df.columns:
                temp_df["currentowner"] = ""

            if "reportedon" not in temp_df.columns:
                temp_df["reportedon"] = ""

            if "closeddate" not in temp_df.columns:
                temp_df["closeddate"] = ""

            if "ageing" not in temp_df.columns:
                temp_df["ageing"] = 0

            # -------------------------
            # 1) Parse dates (CRUCIAL)
            # -------------------------
            temp_df["reportedon"] = pd.to_datetime(temp_df["reportedon"], errors="coerce")
            temp_df["closeddate"] = pd.to_datetime(temp_df["closeddate"], errors="coerce")

            # -------------------------
            # 2) Fix / Infer statuscode
            # -------------------------
            # If Statuscode exists but empty, treat as Pending
            if "statuscode" in temp_df.columns:
                temp_df["statuscode"] = temp_df["statuscode"].fillna("").astype(str).str.strip()
                temp_df.loc[temp_df["statuscode"] == "", "statuscode"] = "Pending"
            elif "Statuscode" in temp_df.columns:
                temp_df["statuscode"] = temp_df["Statuscode"].fillna("").astype(str).str.strip()
                temp_df.loc[temp_df["statuscode"] == "", "statuscode"] = "Pending"
            else:
                temp_df["statuscode"] = "Pending"

            # If closeddate exists => Resolved (strongest truth)
            temp_df.loc[temp_df["closeddate"].notna(), "statuscode"] = "Resolved"

            # -------------------------
            # 3) Correct aging calculation in DAYS
            # -------------------------
            today = pd.Timestamp.now()

            def calc_aging_days(row):
                start = row["reportedon"]
                if pd.isnull(start):
                    return _raw_age_to_days(row.get("ageing", 0))

                end = row["closeddate"] if pd.notnull(row["closeddate"]) else today
                diff_days = (end - start).total_seconds() / 86400.0
                if diff_days < 0:
                    diff_days = 0.0
                return diff_days

            temp_df["aging_num"] = temp_df.apply(calc_aging_days, axis=1)

            _df = temp_df.fillna("")
            print(f"[user_insights] Loaded CSV with encoding: {enc}")
            print(f"[user_insights] Columns: {_df.columns.tolist()}")
            return _df
        except Exception:
            continue

    raise RuntimeError("Failed to load CSV with known encodings.")

# -----------------------------
# Utility
# -----------------------------
def is_case_id(value: str) -> bool:
    return str(value).isdigit()

# -----------------------------
# OWNERID-AWARE FILTER (CORE FIX)
# -----------------------------
def _get_user_cases_by_owner(owner_name: str, df: pd.DataFrame):
    owner_key = owner_name.strip().lower()
    owner_id = OWNER_MAP.get(owner_key)

    if owner_id and "currentownerid" in df.columns:
        return df[df["currentownerid"] == owner_id]

    return df[
        df["currentowner"]
        .astype(str)
        .str.lower()
        .str.strip()
        .eq(owner_key)
    ]

# -----------------------------
# Core Logic Helpers
# -----------------------------
def _get_user_cases(owner_name: str):
    df = load_cases_df()
    return _get_user_cases_by_owner(owner_name, df).copy()

def _get_active_user_cases(owner_name: str):
    user_df = _get_user_cases(owner_name)
    if user_df.empty:
        return pd.DataFrame()

    mask = ~user_df["statuscode"].astype(str).str.strip().isin(TERMINAL_STATUSES)
    return user_df[mask].copy()

# -----------------------------
# Main Insight Function
# -----------------------------
def get_user_or_case_insights(query_val: str):
    df = load_cases_df()
    query_val = str(query_val).strip()

    # Case search
    if is_case_id(query_val):
        case_match = df[df["caseid"].astype(str) == query_val]
        if not case_match.empty:
            return {"type": "case", "data": case_match.iloc[0].to_dict()}

    # User search
    user_match = _get_user_cases(query_val)

    if user_match.empty:
        mask = df["currentowner"].astype(str).str.lower().str.contains(query_val.lower())
        user_match = df[mask].copy()

    if not user_match.empty:
        active_cases_df = _get_active_user_cases(user_match.iloc[0]["currentowner"])

        total = len(user_match)
        pending_count = len(active_cases_df[active_cases_df["aging_num"] <= 7])
        overdue = len(active_cases_df[
            (active_cases_df["aging_num"] > 7) &
            (active_cases_df["aging_num"] <= 21)
        ])
        critical = len(active_cases_df[active_cases_df["aging_num"] > 21])

        status_counts = user_match["statuscode"].value_counts().to_dict()
        primary_owner = user_match.iloc[0]["currentowner"]

        summary = {
            "owner": primary_owner,
            "total_cases": total,
            "pending_cases": pending_count,
            "overdue_cases": overdue,
            "critical_cases": critical,
            "status_breakdown": status_counts
        }
        return {"type": "user", "data": summary}

    return {"type": "none", "data": None}

# -----------------------------
# List Getter Functions
# -----------------------------
def get_pending_cases(owner_name: str, top_n: int = 5):
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty:
        return []

    fresh_df = active_df[active_df["aging_num"] <= 7]
    fresh_df = fresh_df.sort_values(by="aging_num", ascending=False)
    return fresh_df.head(top_n).to_dict(orient="records")

def get_overdue_cases(owner_name: str, top_n: int = 3):
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty:
        return []

    overdue_df = active_df[
        (active_df["aging_num"] > 7) &
        (active_df["aging_num"] <= 21)
    ]
    overdue_df = overdue_df.sort_values(by="aging_num", ascending=False)
    return overdue_df.head(top_n).to_dict(orient="records")

def get_critical_cases(owner_name: str, top_n: int = 3):
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty:
        return []

    critical_df = active_df[active_df["aging_num"] > 21]
    critical_df = critical_df.sort_values(by="aging_num", ascending=False)
    return critical_df.head(top_n).to_dict(orient="records")

def get_recent_cases(owner, days=5):
    df = load_cases_df()
    user_df = _get_user_cases_by_owner(owner, df).copy()

    user_df["reportedon"] = pd.to_datetime(
        user_df["reportedon"], errors="coerce"
    )

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    recent_df = user_df[user_df["reportedon"] >= cutoff]

    return recent_df.to_dict("records")

def get_latest_resolved_cases(owner_name: str, top_n: int = 3):
    df = load_cases_df()
    user_df = _get_user_cases_by_owner(owner_name, df).copy()

    # Ensure datetime types (safe even if already parsed in load_cases_df)
    user_df["reportedon"] = pd.to_datetime(user_df.get("reportedon", ""), errors="coerce")
    user_df["closeddate"] = pd.to_datetime(user_df.get("closeddate", ""), errors="coerce")

    # Resolved definition: statuscode says Resolved OR closeddate exists
    resolved_df = user_df[
        (user_df["statuscode"].astype(str).str.strip() == "Resolved") |
        (user_df["closeddate"].notna())
    ].copy()

    # Keep only rows that truly have a closeddate so sorting is meaningful
    resolved_df = resolved_df[resolved_df["closeddate"].notna()].copy()

    if resolved_df.empty:
        return []

    # Sort by latest closed date
    resolved_df = resolved_df.sort_values(by="closeddate", ascending=False)

    # Compute duration in days
    resolved_df["resolution_days"] = (
        (resolved_df["closeddate"] - resolved_df["reportedon"]).dt.total_seconds() / 86400.0
    ).fillna(0)

    # Ensure effort fields exist + numeric
    for col in ["configurationeffort", "testingeffort", "totaleffort"]:
        if col in resolved_df.columns:
            resolved_df[col] = pd.to_numeric(resolved_df[col], errors="coerce").fillna(0)
        else:
            resolved_df[col] = 0

    return resolved_df.head(top_n).to_dict(orient="records")