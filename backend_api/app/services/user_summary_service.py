import pandas as pd


class UserSummaryService:
    """
    Computes user-level summaries and case drilldowns
    from the master cases CSV.
    """

    # 🔁 Column mappings aligned to NEW schema
    USER_COLUMN = "ownername"
    STATUS_COLUMN = "Statuscode"
    AGING_COLUMN = "ageing"
    CATEGORY_COLUMN = "category"

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def _load_csv(self) -> pd.DataFrame:
        """
        Loads the CSV file using multiple encoding fallbacks.
        """
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

        for encoding in encodings:
            try:
                return pd.read_csv(self.csv_path, encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Unable to read CSV with supported encodings: {self.csv_path}")

    def get_user_rows(self, username: str) -> pd.DataFrame:
        """
        Returns all rows for a given username.
        """
        df = self._load_csv()

        # Normalize owner column + input username
        df[self.USER_COLUMN] = df[self.USER_COLUMN].astype(str).str.lower()
        username = username.lower()

        return df[df[self.USER_COLUMN] == username]

    def compute_user_summary(self, username: str) -> dict:
        """
        Computes summary metrics for a given user.
        """
        user_df = self.get_user_rows(username)

        if user_df.empty:
            return {
                "username": username,
                "total_cases": 0,
                "pending": 0,
                "overdue": 0,
                "critical": 0,
                "status_breakdown": {}
            }

        # Normalize ageing safely
        user_df[self.AGING_COLUMN] = (
            pd.to_numeric(user_df[self.AGING_COLUMN], errors="coerce")
            .fillna(0)
        )

        # Normalize status safely
        user_df[self.STATUS_COLUMN] = (
            user_df[self.STATUS_COLUMN].astype(str).str.lower()
        )

        total_cases = len(user_df)

        # Closed statuses (same logic as earlier)
        CLOSED_STATUSES = {"resolved", "invalid", "closed"}

        open_df = user_df[~user_df[self.STATUS_COLUMN].isin(CLOSED_STATUSES)]

        pending = len(open_df)
        overdue = len(open_df[open_df[self.AGING_COLUMN] > 7])
        critical = len(open_df[open_df[self.AGING_COLUMN] > 21])

        status_breakdown = (
            user_df[self.STATUS_COLUMN]
            .value_counts()
            .to_dict()
        )

        return {
            "username": username,
            "total_cases": total_cases,
            "pending": pending,
            "overdue": overdue,
            "critical": critical,
            "status_breakdown": status_breakdown
        }

    def get_user_cases(self, username: str, case_type: str):
        """
        Returns case-level data for a user based on case type.
        case_type: pending | overdue | critical
        """
        user_df = self.get_user_rows(username)

        if user_df.empty:
            return []

        # Normalize columns
        user_df[self.AGING_COLUMN] = (
            pd.to_numeric(user_df[self.AGING_COLUMN], errors="coerce")
            .fillna(0)
        )

        user_df[self.STATUS_COLUMN] = (
            user_df[self.STATUS_COLUMN].astype(str).str.lower()
        )

        CLOSED_STATUSES = {"resolved", "invalid", "closed"}

        open_df = user_df[~user_df[self.STATUS_COLUMN].isin(CLOSED_STATUSES)]

        if case_type == "pending":
            filtered = open_df
        elif case_type == "overdue":
            filtered = open_df[open_df[self.AGING_COLUMN] > 7]
        elif case_type == "critical":
            filtered = open_df[open_df[self.AGING_COLUMN] > 21]
        else:
            filtered = user_df

        # Return minimal case fields (UI-friendly)
        return filtered[
            [
                "caseid",
                self.STATUS_COLUMN,
                self.AGING_COLUMN,
                self.CATEGORY_COLUMN
            ]
        ].to_dict(orient="records")
