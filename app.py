from __future__ import annotations

import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from backend.predict import predict_customer
from backend.train import train_models
from backend.utils import (
    ARTIFACTS_DIR,
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MLFLOW_DIR,
    MODEL_PACKAGE_PATH,
    ensure_directories,
    load_dataset,
    load_json,
    read_prediction_log,
    resolve_dataset_path,
)


st.set_page_config(page_title="Customer Churn Pipeline", layout="wide", initial_sidebar_state="expanded")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background: #ffffff; color: #111827; }
        section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e5e7eb; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { letter-spacing: 0; color: #111827; }
        div[data-testid="stMetric"] {
            background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 1rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .card {
            background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 1.1rem 1.2rem; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .status-pill {
            display: inline-block; border-radius: 999px; padding: 0.25rem 0.65rem;
            background: #eff6ff; color: #1d4ed8; font-size: 0.85rem; font-weight: 600;
            border: 1px solid #bfdbfe;
        }
        div.stButton > button {
            border-radius: 10px; border: 1px solid #2563eb; background: #2563eb;
            color: #ffffff; font-weight: 600;
        }
        div.stButton > button:hover { border-color: #1d4ed8; background: #1d4ed8; color: #ffffff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


def ensure_model() -> dict:
    ensure_directories()
    if not MODEL_PACKAGE_PATH.exists():
        with st.spinner("Training initial model..."):
            train_models()
    return joblib.load(MODEL_PACKAGE_PATH)


def page_header(title: str, subtitle: str) -> None:
    st.markdown("<span class='status-pill'>ML Portfolio Project</span>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def home_page() -> None:
    package = ensure_model()
    metrics = load_json(METRICS_PATH, default=package.get("metrics", {}))
    df = cached_dataset()
    page_header(
        "Customer Churn Prediction Pipeline",
        "A modular ML workflow with preprocessing, feature engineering, experiment tracking, Airflow automation, and a Streamlit dashboard.",
    )
    cols = st.columns(4)
    cols[0].metric("Best Model", package.get("model_name", "Unknown"))
    cols[1].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    cols[2].metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    cols[3].metric("Customers", f"{len(df):,}")

    st.markdown(
        """
        <div class="card">
        <h3>Project Workflow</h3>
        <p>The pipeline reads a local Telco churn CSV, engineers customer behavior features, fits reusable preprocessing, compares Logistic Regression and Random Forest, logs MLflow runs, and saves the best model package for predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#d1d5db", fillcolor="#f9fafb", fontname="Inter"];
          Data -> Features -> Preprocessing -> Training -> Evaluation -> Dashboard;
          Training -> MLflow;
          Airflow -> Training;
          Dashboard -> Monitoring;
        }
        """
    )
    st.info(f"Active dataset: {resolve_dataset_path(DATA_PATH)}")


def customer_prediction_page() -> None:
    ensure_model()
    page_header("Customer Prediction", "Enter customer attributes and estimate churn probability.")
    with st.form("customer_form"):
        left, middle, right = st.columns(3)
        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with middle:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        with right:
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        lower_left, lower_right = st.columns(2)
        with lower_left:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
        with lower_right:
            monthly = st.number_input("Monthly Charges", min_value=0.0, value=75.0, step=1.0)
            total = st.number_input("Total Charges", min_value=0.0, value=monthly * max(tenure, 1), step=10.0)
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
        }
        result = predict_customer(customer)
        cols = st.columns(3)
        cols[0].metric("Prediction", result["prediction"])
        cols[1].metric("Churn Probability", f"{result['churn_probability']:.1%}")
        cols[2].metric("Confidence", f"{result['confidence']:.1%}")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["churn_probability"] * 100,
            title={"text": "Churn Risk"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563eb"}},
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)


def model_performance_page() -> None:
    package = ensure_model()
    metrics = load_json(METRICS_PATH, default=package.get("metrics", {}))
    page_header("Model Performance", "Evaluate model quality using classification metrics and diagnostics.")

    cols = st.columns(5)
    for col, key in zip(cols, ["accuracy", "precision", "recall", "f1_score", "roc_auc"]):
        col.metric(key.replace("_", " ").title(), f"{metrics.get(key, 0):.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        path = ARTIFACTS_DIR / "confusion_matrix.png"
        if path.exists():
            st.image(str(path), use_container_width=True)
    with right:
        st.subheader("ROC Curve")
        path = ARTIFACTS_DIR / "roc_curve.png"
        if path.exists():
            st.image(str(path), use_container_width=True)

    if FEATURE_IMPORTANCE_PATH.exists():
        importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        st.subheader("Feature Importance")
        fig = px.bar(
            importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            color_discrete_sequence=["#2563eb"],
        )
        fig.update_layout(height=520, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)


def dataset_overview_page() -> None:
    df = cached_dataset()
    page_header("Dataset Overview", "Profile the local Telco churn dataset used for training.")
    churn_rate = df["Churn"].eq("Yes").mean() * 100 if "Churn" in df.columns else 0
    missing_values = int(df.isna().sum().sum())

    cols = st.columns(4)
    cols[0].metric("Customers", f"{len(df):,}")
    cols[1].metric("Columns", f"{df.shape[1]:,}")
    cols[2].metric("Churn Rate", f"{churn_rate:.1f}%")
    cols[3].metric("Missing Values", f"{missing_values:,}")

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    left, right = st.columns(2)
    with left:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(churn_counts, names="Churn", values="Count", hole=0.55, color_discrete_sequence=["#2563eb", "#94a3b8"])
        fig.update_layout(title="Churn Distribution", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(df, x="MonthlyCharges", color="Churn", barmode="overlay", color_discrete_sequence=["#2563eb", "#64748b"])
        fig.update_layout(title="Monthly Charges by Churn", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missing Values by Column")
    missing = df.isna().sum().reset_index()
    missing.columns = ["Column", "Missing"]
    st.dataframe(missing[missing["Missing"] > 0], use_container_width=True)


def experiment_tracking_page() -> None:
    ensure_model()
    page_header("Experiment Tracking", "Inspect recent MLflow runs and the best model metrics.")
    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    runs = mlflow.search_runs(experiment_names=["customer-churn-prediction"], order_by=["metrics.roc_auc DESC"])

    if runs.empty:
        st.info("No MLflow runs found yet. Train the model to populate experiment history.")
        return

    visible = [
        "run_id", "tags.mlflow.runName", "metrics.roc_auc", "metrics.f1_score",
        "metrics.accuracy", "metrics.precision", "metrics.recall", "start_time",
    ]
    existing = [col for col in visible if col in runs.columns]
    st.dataframe(runs[existing], use_container_width=True)

    best = runs.iloc[0]
    cols = st.columns(4)
    cols[0].metric("Best Run", best.get("tags.mlflow.runName", "Unknown"))
    cols[1].metric("Best ROC-AUC", f"{best.get('metrics.roc_auc', 0):.3f}")
    cols[2].metric("Best F1", f"{best.get('metrics.f1_score', 0):.3f}")
    cols[3].metric("Total Runs", f"{len(runs):,}")


def monitoring_page() -> None:
    package = ensure_model()
    logs = read_prediction_log()
    page_header("Monitoring", "Simple local monitoring for dashboard prediction activity.")
    last_retraining = (
        pd.to_datetime(MODEL_PACKAGE_PATH.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M:%S")
        if MODEL_PACKAGE_PATH.exists()
        else "Not trained"
    )

    cols = st.columns(4)
    cols[0].metric("Predictions Made", f"{len(logs):,}")
    cols[1].metric("Avg Confidence", f"{logs['confidence'].mean():.1%}" if not logs.empty else "0.0%")
    cols[2].metric("Last Retraining", last_retraining)
    cols[3].metric("Active Model", package.get("model_name", "Unknown"))

    if logs.empty:
        st.info("No predictions have been logged yet. Use the Customer Prediction page to create monitoring data.")
        return

    left, right = st.columns(2)
    with left:
        counts = logs["prediction"].value_counts().reset_index()
        counts.columns = ["Prediction", "Count"]
        fig = px.bar(counts, x="Prediction", y="Count", color_discrete_sequence=["#2563eb"])
        fig.update_layout(title="Prediction Distribution", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.line(logs, x="timestamp", y="churn_probability", markers=True)
        fig.update_traces(line_color="#2563eb", marker_color="#2563eb")
        fig.update_layout(title="Churn Probability Over Time", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
        st.plotly_chart(fig, use_container_width=True)


def airflow_page() -> None:
    page_header("Airflow Workflow", "A lightweight DAG automates the local retraining flow.")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR;
          node [shape=box, style="rounded,filled", color="#d1d5db", fillcolor="#f9fafb", fontname="Inter"];
          load_data -> preprocess_data -> train_model -> evaluate_model -> save_model;
        }
        """
    )
    st.markdown(
        """
        <div class="card">
        <h3>Local DAG</h3>
        <p>The DAG in <code>airflow/dags/churn_retraining_dag.py</code> calls the same backend training function used by the command line and dashboard, keeping scheduled retraining consistent with manual retraining.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Run retraining now"):
        with st.spinner("Training models and refreshing artifacts..."):
            result = train_models()
        st.success(f"Retraining complete. Best model: {result.model_name} with ROC-AUC {result.metrics['roc_auc']:.3f}")


def main() -> None:
    inject_css()
    st.sidebar.title("Churn Pipeline")
    st.sidebar.caption("Machine learning portfolio app")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Customer Prediction",
            "Model Performance",
            "Dataset Overview",
            "Experiment Tracking",
            "Monitoring",
            "Airflow Workflow",
        ],
    )
    pages = {
        "Home": home_page,
        "Customer Prediction": customer_prediction_page,
        "Model Performance": model_performance_page,
        "Dataset Overview": dataset_overview_page,
        "Experiment Tracking": experiment_tracking_page,
        "Monitoring": monitoring_page,
        "Airflow Workflow": airflow_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
