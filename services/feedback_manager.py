import os
import pandas as pd
from datetime import datetime
import numpy as np

FEEDBACK_FILE = "data/feedback_log.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "mpr_subject",
    "similarity_score",
    "quality_rating",
    "relevance_rating",
    "clarity_rating",
    "match_accuracy",
    "applicability",
    "final_reward"
]


# =========================
# File Initialization
# =========================
def _initialize_file():
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        df.to_csv(FEEDBACK_FILE, index=False)


# =========================
# Reward Calculation
# =========================
def compute_reward(row):
    match_score = 1 if row["match_accuracy"] == "Yes" else 0

    applicability_score = {
        "Immediate": 1,
        "Needs Customization": 0.5,
        "Not Useful": 0
    }.get(row["applicability"], 0)

    reward = (
        0.3 * row["quality_rating"] +
        0.25 * row["relevance_rating"] +
        0.2 * row["clarity_rating"] +
        0.15 * match_score +
        0.1 * applicability_score
    )

    return round(reward, 3)


# =========================
# Save Feedback
# =========================
def save_feedback(data: dict):
    _initialize_file()

    data["timestamp"] = datetime.utcnow()
    reward = compute_reward(data)
    data["final_reward"] = reward

    df = pd.read_csv(FEEDBACK_FILE)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    return reward


# =========================
# Load Feedback
# =========================
def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(FEEDBACK_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# =========================
# Recency Weighting
# =========================
def apply_recency_weighting(df, decay_lambda=0.01):
    if df.empty:
        return df

    # FIX: Ensure 'now' and 'timestamp' are both UTC aware
    now = pd.Timestamp.utcnow()
    
    # Check if the column already has timezone info; if not, localize to UTC
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize('UTC')
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert('UTC')

    df["age_days"] = (now - df["timestamp"]).dt.days
    df["recency_weight"] = np.exp(-decay_lambda * df["age_days"])

    df["weighted_reward"] = df["final_reward"] * df["recency_weight"]

    return df


# =========================
# Recency-Aware Subject Score
# =========================
def get_subject_success_rate(mpr_subject, decay_lambda=0.01):

    df = load_feedback()

    if df.empty:
        return 3.0

    df = apply_recency_weighting(df, decay_lambda)

    subject_df = df[df["mpr_subject"] == mpr_subject]

    if subject_df.empty:
        return 3.0

    weighted_sum = subject_df["weighted_reward"].sum()
    weight_total = subject_df["recency_weight"].sum()

    return round(weighted_sum / weight_total, 3) if weight_total > 0 else 3.0


# =========================
# Subject Feedback Count
# =========================
def get_subject_feedback_count(mpr_subject):
    df = load_feedback()
    return len(df[df["mpr_subject"] == mpr_subject])

# =========================
# Recency Weighted Subject Score
# =========================
def get_weighted_subject_score(mpr_subject, decay_lambda=0.01):
    """
    Returns recency-weighted average score
    for a given subject.
    """

    df = load_feedback()

    if df.empty:
        return 0

    df = df[df["mpr_subject"] == mpr_subject]

    if df.empty:
        return 0

    df = apply_recency_weighting(df, decay_lambda)

    weighted_score = df["weighted_reward"].sum()
    total_weight = df["recency_weight"].sum()

    if total_weight == 0:
        return 0

    return round(weighted_score / total_weight, 3)



# =========================
# Global Feedback Stats
# =========================
def get_feedback_stats(decay_lambda=0.01):

    df = load_feedback()

    if df.empty:
        return {
            "total_feedback": 0,
            "global_avg_reward": 0,
            "reward_std": 0,
            "learning_velocity": 0,
            "raw_df": df
        }

    df = apply_recency_weighting(df, decay_lambda)

    weighted_sum = df["weighted_reward"].sum()
    weight_total = df["recency_weight"].sum()

    global_avg = weighted_sum / weight_total if weight_total > 0 else 0
    reward_std = df["final_reward"].std()

    df = df.sort_values("timestamp")

    window = min(10, len(df))
    df["rolling_avg"] = df["final_reward"].rolling(window=window).mean()
    df["delta"] = df["rolling_avg"].diff()

    learning_velocity = df["delta"].mean()

    return {
        "total_feedback": len(df),
        "global_avg_reward": round(global_avg, 3),
        "reward_std": round(reward_std, 3),
        "learning_velocity": round(learning_velocity, 4),
        "raw_df": df
    }


# =========================
# Reward Trend
# =========================
def get_reward_trend(decay_lambda=0.01):

    df = load_feedback()

    if df.empty:
        return pd.DataFrame()

    df = apply_recency_weighting(df, decay_lambda)
    df = df.sort_values("timestamp")

    window = min(10, len(df))
    df["rolling_avg"] = df["weighted_reward"].rolling(window=window).mean()

    return df[["timestamp", "rolling_avg"]]

# =========================
# Low Performing Subjects (Recency Aware)
# =========================
def get_low_performing_subjects(feedback_df=None, threshold=2.5, min_samples=3):
    """
    Returns subjects whose average reward falls below threshold.
    Auto-loads data if feedback_df is not provided.
    """
    
    # FIX: Auto-load if called from retriever without arguments
    if feedback_df is None:
        feedback_df = load_feedback()

    if feedback_df.empty:
        return set()

    subject_perf = (
        feedback_df.groupby("mpr_subject")
        .agg(
            avg_reward=("final_reward", "mean"), # Ensure using 'final_reward'
            count=("final_reward", "count")
        )
        .reset_index()
    )

    low = subject_perf[
        (subject_perf["avg_reward"] < threshold) &
        (subject_perf["count"] >= min_samples)
    ]

    return set(low["mpr_subject"].tolist())

# =================================
# ==========Calibration curve==================
# =====================================

def compute_confidence_calibration(feedback_df, bins=5):
    """
    Compare predicted similarity vs actual reward.
    Returns dataframe for calibration curve.
    """
    import pandas as pd
    import numpy as np

    if feedback_df.empty:
        return pd.DataFrame()

    df = feedback_df.copy()

    # 1. Normalize similarity to 0-1
    df["similarity_norm"] = df["similarity_score"] / 100

    # 2. Bin predictions
    df["confidence_bin"] = pd.cut(
        df["similarity_norm"],
        bins=bins,
        labels=False
    )

    # 3. Perform Aggregation (CRITICAL: Every key must match your CSV columns)
    # Your CSV uses 'final_reward', NOT 'reward'
    calibration = (
        df.groupby("confidence_bin")
        .agg(
            avg_predicted=("similarity_norm", "mean"),
            avg_actual_reward=("final_reward", "mean"),
            count=("final_reward", "count")
        )
        .reset_index()
    )

    return calibration

def check_retrain_trigger(
    min_feedback=50,
    stability_threshold=1.2
):
    """
    Determines whether model retraining is recommended.
    """

    stats = get_feedback_stats()

    total_feedback = stats.get("total_feedback", 0)
    model_stability = stats.get("reward_std", 0)

    trigger = (
        total_feedback >= min_feedback and
        model_stability >= stability_threshold
    )

    return {
        "trigger": trigger,
        "total_feedback": total_feedback,
        "model_stability": model_stability,
        "min_required": min_feedback,
        "stability_threshold": stability_threshold
    }


