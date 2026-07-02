from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.utils import ARTIFACTS_DIR, FEATURE_IMPORTANCE_PATH, METRICS_PATH, save_json


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray) -> dict[str, float]:
    """Calculate standard binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
    }


def save_evaluation_artifacts(
    model: Any,
    x_test_processed: np.ndarray,
    y_test: pd.Series,
    feature_names: list[str],
    metrics: dict[str, float],
    output_dir: Path = ARTIFACTS_DIR,
) -> None:
    """Save metrics, confusion matrix, ROC curve, and feature importance artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = model.predict(x_test_processed)
    y_probability = model.predict_proba(x_test_processed)[:, 1]

    save_json(metrics, METRICS_PATH)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_probability, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    importance = extract_feature_importance(model, feature_names)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importance for tree models or absolute coefficients for linear models."""
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_[0])
    else:
        scores = np.zeros(len(feature_names))

    return (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(20)
    )

