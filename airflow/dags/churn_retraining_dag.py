from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

default_args = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="customer_churn_retraining",
    description="Load local churn data, preprocess, train, evaluate, and save the best model.",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["machine-learning", "churn", "portfolio"],
)
def customer_churn_retraining_dag():
    @task
    def load_data() -> int:
        from backend.utils import DATA_PATH, load_dataset, resolve_dataset_path

        df = load_dataset(DATA_PATH)
        print(f"Loaded {len(df)} rows from {resolve_dataset_path(DATA_PATH)}")
        return len(df)

    @task
    def preprocess_data(row_count: int) -> str:
        from backend.preprocessing import preprocess_for_training
        from backend.utils import DATA_PATH, load_dataset

        df = load_dataset(DATA_PATH)
        preprocess_for_training(df)
        return f"Preprocessing validated for {row_count} rows."

    @task
    def train_model(_: str) -> dict[str, float | str]:
        from backend.train import train_models

        result = train_models()
        return {"model_name": result.model_name, **result.metrics}

    @task
    def evaluate_model(training_result: dict[str, float | str]) -> dict[str, float | str]:
        print(f"Evaluation metrics: {training_result}")
        return training_result

    @task
    def save_model(evaluation_result: dict[str, float | str]) -> str:
        from backend.utils import MODEL_PACKAGE_PATH

        message = f"Saved {evaluation_result['model_name']} to {MODEL_PACKAGE_PATH}"
        print(message)
        return message

    rows = load_data()
    preprocessed = preprocess_data(rows)
    trained = train_model(preprocessed)
    evaluated = evaluate_model(trained)
    save_model(evaluated)


customer_churn_retraining_dag()

