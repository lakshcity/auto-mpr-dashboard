import os
import pandas as pd
from datetime import datetime

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


def _initialize_file():
    """Create feedback file if it doesn't exist."""
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        df.to_csv(FEEDBACK_FILE, index=False)


def compute_reward(row):
    """
    Weighted reward formula.
    Adjust weights later if needed.
    """
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


def save_feedback(data: dict):
    """
    Save structured feedback entry.
    """
    _initialize_file()

    data["timestamp"] = datetime.utcnow()

    reward = compute_reward(data)
    data["final_reward"] = reward

    df = pd.read_csv(FEEDBACK_FILE)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    return reward


def get_subject_success_rate(mpr_subject):
    """
    Average reward score for a subject.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return 0.0

    df = pd.read_csv(FEEDBACK_FILE)
    subject_df = df[df["mpr_subject"] == mpr_subject]

    if subject_df.empty:
        return 0.0

    return round(subject_df["final_reward"].mean(), 3)


def get_feedback_stats():
    """
    Global feedback insights.
    """
    if not os.path.exists(FEEDBACK_FILE):
        return {}

    df = pd.read_csv(FEEDBACK_FILE)

    return {
        "total_feedback": len(df),
        "avg_reward": round(df["final_reward"].mean(), 3),
        "best_subjects": df.groupby("mpr_subject")["final_reward"]
                           .mean()
                           .sort_values(ascending=False)
                           .head(5)
                           .to_dict()
    }
