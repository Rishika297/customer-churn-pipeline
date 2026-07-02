from __future__ import annotations

import numpy as np
import pandas as pd


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple, interpretable churn features from raw Telco-style data."""
    data = df.copy()

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    if "tenure" in data.columns:
        tenure = data["tenure"].fillna(0).clip(lower=0)
        data["TenureGroup"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12", "13-24", "25-48", "49+"],
        ).astype(str)

    if {"TotalCharges", "tenure"}.issubset(data.columns):
        safe_tenure = data["tenure"].fillna(0).clip(lower=1)
        data["AvgMonthlySpend"] = data["TotalCharges"] / safe_tenure
        data["TotalChargesPerMonth"] = data["AvgMonthlySpend"]

    available_service_columns = [col for col in SERVICE_COLUMNS if col in data.columns]
    if available_service_columns:
        data["NumServices"] = data[available_service_columns].apply(
            lambda row: sum(str(value).strip().lower() == "yes" for value in row),
            axis=1,
        )

    if "Contract" in data.columns:
        data["IsMonthToMonth"] = (data["Contract"] == "Month-to-month").astype(int)
        data["HasLongTermContract"] = data["Contract"].isin(["One year", "Two year"]).astype(int)

    if "InternetService" in data.columns:
        data["HasFiberOptic"] = (data["InternetService"] == "Fiber optic").astype(int)
        data["HasInternetService"] = (data["InternetService"] != "No").astype(int)

    return data


def prepare_target(df: pd.DataFrame, target_column: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    """Split features and target, mapping Yes/No churn labels to 1/0."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    features = df.drop(columns=[target_column])
    if "customerID" in features.columns:
        features = features.drop(columns=["customerID"])

    target = df[target_column].map({"No": 0, "Yes": 1})
    if target.isna().any():
        target = df[target_column].astype(int)

    return features, target.astype(int)

