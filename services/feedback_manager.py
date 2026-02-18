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

def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = pd.read_csv(FEEDBACK_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

def get_feedback_stats():

    df = load_feedback()

    if df.empty:
        return {
            "total_feedback": 0,
            "global_avg_reward": 0,
            "reward_std": 0,
            "learning_velocity": 0,
            "raw_df": df
        }

    global_avg = df["final_reward"].mean()
    reward_std = df["final_reward"].std()

    df = df.sort_values("timestamp")

    df["rolling_avg"] = df["final_reward"].rolling(
        window=min(10, len(df))
    ).mean()

    df["delta"] = df["rolling_avg"].diff()
    learning_velocity = df["delta"].mean()

    return {
        "total_feedback": len(df),
        "global_avg_reward": round(global_avg, 3),
        "reward_std": round(reward_std, 3),
        "learning_velocity": round(learning_velocity, 4),
        "raw_df": df
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

def get_weighted_subject_score(subject, min_samples=5):
    df = load_feedback()

    subject_df = df[df["mpr_subject"] == subject]

    n = len(subject_df)
    if n == 0:
        return 3.0  # neutral default

    avg = subject_df["quality_rating"].mean()

    # Bayesian smoothing
    global_avg = df["quality_rating"].mean()
    weight = n / (n + min_samples)

    return round(weight * avg + (1 - weight) * global_avg, 2)

def get_low_performing_subjects(threshold=2.0, min_samples=3):

    df = load_feedback()

    if df.empty:
        return []

    subject_stats = (
        df.groupby("mpr_subject")
        .agg(
            count=("quality_rating", "count"),
            avg_reward=("final_reward", "mean")
        )
        .reset_index()
    )

    low_subjects = subject_stats[
        (subject_stats["count"] >= min_samples) &
        (subject_stats["avg_reward"] < threshold)
    ]

    return low_subjects["mpr_subject"].tolist()


