from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.feature_engineering import add_customer_features, prepare_target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline with imputation, scaling, and one-hot encoding."""
    numeric_features = features.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply feature engineering and create a stratified train/test split."""
    engineered = add_customer_features(df)
    features, target = prepare_target(engineered)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def preprocess_for_training(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return raw splits plus fitted preprocessing artifacts for model training."""
    x_train, x_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)
    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_processed": x_train_processed,
        "x_test_processed": x_test_processed,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.get_feature_names_out().tolist(),
    }

