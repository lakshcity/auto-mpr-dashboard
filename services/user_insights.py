import pandas as pd
from pathlib import Path

# -----------------------------
# Configuration: Business Logic
# -----------------------------
# Statuses that mean the work is finished (Excluded from Aging counts)
TERMINAL_STATUSES = ["Resolved", "Closed", "Invalid"]

# -----------------------------
# Helpers
# -----------------------------
def _to_int(val):
    try:
        # Handle "3 D" or "3" or 3
        if isinstance(val, str):
            clean_val = val.lower().replace("d", "").strip()
            return int(clean_val)
        return int(val)
    except Exception:
        return 0

# -----------------------------
# Load data ONCE (cached)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
# FIX 1: Point to the new Master file
DATA_PATH = BASE_DIR / "app" / "data" / "cases_master.csv"

_df = None

def load_cases_df():
    global _df
    if _df is not None:
        return _df

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            temp_df = pd.read_csv(DATA_PATH, encoding=enc)
            
            # FIX 2: Standardize Column Names (New Data -> Old Logic)
            column_mapping = {
                "ownername": "currentowner",  # Map new 'ownername' to expected 'currentowner'
                "ageing": "aging",            # Map new 'ageing' to expected 'aging'
                "Statuscode": "statuscode",   # Map new 'Statuscode' to lowercase
                "MPR_Subject": "subject",     # Fallback if 'subject' is missing
            }
            temp_df.rename(columns=column_mapping, inplace=True)
            
            # FIX 3: Ensure critical columns exist
            if "statuscode" not in temp_df.columns:
                temp_df["statuscode"] = "Unknown"
            else:
                temp_df["statuscode"] = temp_df["statuscode"].fillna("Unknown")

            # FIX 4: Create a clean numeric aging column for charts
            temp_df["aging_num"] = temp_df["aging"].apply(_to_int)

            _df = temp_df.fillna("")
            print(f"[user_insights] Loaded Master CSV with encoding: {enc}")
            print(f"[user_insights] Columns found: {_df.columns.tolist()}")
            return _df
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Failed to load CSV with known encodings. Last error: {last_err}"
    )

# -----------------------------
# Utility: detect caseid vs username
# -----------------------------
def is_case_id(value: str) -> bool:
    # Check if value is digits (Case ID) or String (User Name)
    return str(value).isdigit()

# -----------------------------
# Core Logic Helpers
# -----------------------------
def _get_user_cases(owner_name: str):
    """Get ALL cases for a user (History + Active)"""
    df = load_cases_df()
    # Case-insensitive match for owner name
    mask = df["currentowner"].astype(str).str.lower() == owner_name.lower().strip()
    return df[mask].copy()

def _get_active_user_cases(owner_name: str):
    """Get ONLY Active cases (removing Resolved/Closed)"""
    user_df = _get_user_cases(owner_name)
    if user_df.empty:
        return pd.DataFrame()
    
    # Filter out Terminal Statuses
    mask = ~user_df["statuscode"].astype(str).str.strip().isin(TERMINAL_STATUSES)
    return user_df[mask].copy()

# -----------------------------
# Main Insight Function
# -----------------------------
def get_user_or_case_insights(query_val: str):
    df = load_cases_df()
    query_val = str(query_val).strip()

    # A. Search by Case ID
    if is_case_id(query_val):
        case_match = df[df["caseid"].astype(str) == query_val]
        if not case_match.empty:
            return {"type": "case", "data": case_match.iloc[0].to_dict()}
    
    # B. Search by User (Owner)
    # Try exact match first, then partial
    user_match = _get_user_cases(query_val)
    
    if user_match.empty:
        # Try finding owner containing the string
        mask = df["currentowner"].astype(str).str.lower().str.contains(query_val.lower())
        user_match = df[mask].copy()

    if not user_match.empty:
        # === LOGIC CORRECTION (Mutually Exclusive Buckets) ===
        
        # 1. Get only ACTIVE cases for aging calculations
        active_cases_df = _get_active_user_cases(user_match.iloc[0]["currentowner"])
        
        # 2. Total is strictly history
        total = len(user_match)
        
        # 3. Calculate Buckets (Active Only)
        # Pending (Fresh) = 0 to 7 days
        pending_count = len(active_cases_df[active_cases_df["aging_num"] <= 7])
        
        # Overdue = 8 to 21 days (Strict)
        overdue = len(active_cases_df[
            (active_cases_df["aging_num"] > 7) & 
            (active_cases_df["aging_num"] <= 21)
        ])
        
        # Critical = > 21 days
        critical = len(active_cases_df[active_cases_df["aging_num"] > 21])
        
        # 4. Status Breakdown (for the Pie Chart)
        status_counts = user_match["statuscode"].value_counts().to_dict()

        primary_owner = user_match.iloc[0]["currentowner"]

        summary = {
            "owner": primary_owner,
            "total_cases": total,
            "pending_cases": pending_count, # Means "Fresh (<7d)"
            "overdue_cases": overdue,       # Means "8-21d"
            "critical_cases": critical,     # Means ">21d"
            "status_breakdown": status_counts
        }
        return {"type": "user", "data": summary}

    return {"type": "none", "data": None}

# -----------------------------
# List Getter Functions (Fixed Logic)
# -----------------------------

def get_pending_cases(owner_name: str, top_n: int = 5):
    """
    Returns 'Fresh' active cases (<= 7 days).
    """
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty: return []

    # Filter <= 7
    fresh_df = active_df[active_df["aging_num"] <= 7]
    
    # Sort by Age (Oldest first)
    fresh_df = fresh_df.sort_values(by="aging_num", ascending=False)
    
    return fresh_df.head(top_n).to_dict(orient="records")


def get_overdue_cases(owner_name: str, top_n: int = 3):
    """
    Returns 'Overdue' active cases (8 to 21 days).
    """
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty: return []

    # Filter 8-21
    overdue_df = active_df[
        (active_df["aging_num"] > 7) & 
        (active_df["aging_num"] <= 21)
    ]
    
    # Sort by Age
    overdue_df = overdue_df.sort_values(by="aging_num", ascending=False)
    
    return overdue_df.head(top_n).to_dict(orient="records")


def get_critical_cases(owner_name: str, top_n: int = 3):
    """
    Returns 'Critical' active cases (> 21 days).
    """
    active_df = _get_active_user_cases(owner_name)
    if active_df.empty: return []

    # Filter > 21
    critical_df = active_df[active_df["aging_num"] > 21]
    
    # Sort by Age
    critical_df = critical_df.sort_values(by="aging_num", ascending=False)
    
    return critical_df.head(top_n).to_dict(orient="records")