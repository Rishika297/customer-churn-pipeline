from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import pandas as pd

from backend.feature_engineering import add_customer_features
from backend.utils import MODEL_PACKAGE_PATH, append_prediction_log


def load_model_package(path: str = str(MODEL_PACKAGE_PATH)) -> dict[str, Any]:
    """Load the saved model and preprocessing pipeline."""
    return joblib.load(path)


def predict_customer(customer_data: dict[str, Any], log_prediction: bool = True) -> dict[str, Any]:
    """Predict churn for one customer dictionary."""
    package = load_model_package()
    input_df = pd.DataFrame([customer_data])
    engineered = add_customer_features(input_df)
    model_features = engineered.reindex(columns=package["raw_feature_columns"], fill_value=None)
    processed = package["preprocessor"].transform(model_features)

    probability = float(package["model"].predict_proba(processed)[0, 1])
    prediction = int(probability >= 0.5)
    confidence = probability if prediction == 1 else 1 - probability
    result = {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "prediction_label": prediction,
        "churn_probability": probability,
        "confidence": float(confidence),
        "model_name": package.get("model_name", "Unknown"),
    }

    if log_prediction:
        append_prediction_log(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "prediction": result["prediction"],
                "churn_probability": round(probability, 6),
                "confidence": round(confidence, 6),
            }
        )

    return result

