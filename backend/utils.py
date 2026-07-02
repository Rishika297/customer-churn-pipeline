from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DIR = PROJECT_ROOT / "mlflow"
PREDICTION_LOG_PATH = ARTIFACTS_DIR / "prediction_log.csv"
MODEL_PACKAGE_PATH = MODELS_DIR / "best_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"


def ensure_directories() -> None:
    """Create project output directories used by training and the dashboard."""
    for path in (ARTIFACTS_DIR, MODELS_DIR, MLFLOW_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_dataset_path(path: Path | str = DATA_PATH) -> Path:
    """Find a local Telco churn CSV, preferring a Kaggle upload when present."""
    csv_path = Path(path)
    if csv_path.exists() and csv_path.name != "telco_churn.csv":
        return csv_path

    kaggle_matches = sorted(DATA_DIR.rglob("*Telco*Customer*Churn*.csv"))
    if kaggle_matches:
        return kaggle_matches[0]

    churn_matches = sorted(DATA_DIR.rglob("*churn*.csv"))
    non_sample_matches = [match for match in churn_matches if match.name != "telco_churn.csv"]
    if non_sample_matches:
        return non_sample_matches[0]

    if csv_path.exists():
        return csv_path

    raise FileNotFoundError(
        f"Dataset not found. Add the Kaggle Telco churn CSV under {DATA_DIR} or keep {DATA_PATH}."
    )


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the local churn CSV file."""
    csv_path = resolve_dataset_path(path)
    return pd.read_csv(csv_path)


def save_json(data: dict[str, Any], path: Path | str) -> None:
    """Save a dictionary as pretty JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: Path | str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load JSON data, returning a default value when the file does not exist."""
    input_path = Path(path)
    if not input_path.exists():
        return default or {}
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def append_prediction_log(record: dict[str, Any]) -> None:
    """Append a single prediction event to the monitoring log."""
    ensure_directories()
    new_row = pd.DataFrame([record])
    if PREDICTION_LOG_PATH.exists():
        existing = pd.read_csv(PREDICTION_LOG_PATH)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(PREDICTION_LOG_PATH, index=False)


def read_prediction_log() -> pd.DataFrame:
    """Read logged predictions for the monitoring dashboard."""
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp", "prediction", "churn_probability", "confidence"]
        )
    return pd.read_csv(PREDICTION_LOG_PATH)
