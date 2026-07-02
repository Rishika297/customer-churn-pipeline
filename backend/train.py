from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from backend.evaluate import calculate_metrics, save_evaluation_artifacts
from backend.preprocessing import preprocess_for_training
from backend.utils import DATA_PATH, MLFLOW_DIR, MODEL_PACKAGE_PATH, ensure_directories, load_dataset


@dataclass
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    model: Any


def get_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Create candidate churn classifiers."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def train_models(random_state: int = 42) -> TrainingResult:
    """Train candidate models, log MLflow runs, and persist the best model package."""
    ensure_directories()
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment("customer-churn-prediction")

    df = load_dataset(DATA_PATH)
    data = preprocess_for_training(df, random_state=random_state)
    best_result: TrainingResult | None = None

    for model_name, model in get_candidate_models(random_state).items():
        with mlflow.start_run(run_name=model_name):
            model.fit(data["x_train_processed"], data["y_train"])
            y_pred = model.predict(data["x_test_processed"])
            y_probability = model.predict_proba(data["x_test_processed"])[:, 1]
            metrics = calculate_metrics(data["y_test"], y_pred, y_probability)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            result = TrainingResult(model_name=model_name, metrics=metrics, model=model)
            if best_result is None or metrics["roc_auc"] > best_result.metrics["roc_auc"]:
                best_result = result

    if best_result is None:
        raise RuntimeError("No model was trained.")

    package = {
        "model": best_result.model,
        "preprocessor": data["preprocessor"],
        "feature_names": data["feature_names"],
        "raw_feature_columns": data["x_train"].columns.tolist(),
        "model_name": best_result.model_name,
        "metrics": best_result.metrics,
    }
    joblib.dump(package, MODEL_PACKAGE_PATH)
    save_evaluation_artifacts(
        best_result.model,
        data["x_test_processed"],
        data["y_test"],
        data["feature_names"],
        best_result.metrics,
    )
    return best_result


def main() -> None:
    result = train_models()
    print(f"Best model: {result.model_name}")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

