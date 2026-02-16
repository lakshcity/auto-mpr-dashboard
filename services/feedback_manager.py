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
    if not os.path.exists(FEEDBACK_FILE):
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        df.to_csv(FEEDBACK_FILE, index=False)


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


def save_feedback(data: dict):
    _initialize_file()

    data["timestamp"] = datetime.utcnow()

    reward = compute_reward(data)
    data["final_reward"] = reward

    df = pd.read_csv(FEEDBACK_FILE)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    return reward


def get_subject_success_rate(mpr_subject):
    if not os.path.exists(FEEDBACK_FILE):
        return 0.0

    df = pd.read_csv(FEEDBACK_FILE)
    subject_df = df[df["mpr_subject"] == mpr_subject]

    if subject_df.empty:
        return 0.0

    return round(subject_df["final_reward"].mean(), 3)


def get_subject_feedback_count(mpr_subject):
    if not os.path.exists(FEEDBACK_FILE):
        return 0

    df = pd.read_csv(FEEDBACK_FILE)
    return len(df[df["mpr_subject"] == mpr_subject])


def get_feedback_stats():
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


def get_reward_trend():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame()

    df = pd.read_csv(FEEDBACK_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    window_size = min(10, len(df))
    df["rolling_avg"] = df["final_reward"].rolling(window=window_size).mean()


    return df
