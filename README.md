# Customer Churn Prediction Pipeline

A clean, portfolio-ready machine learning project that predicts whether a telecom customer is likely to churn. The project demonstrates preprocessing, feature engineering, model training, evaluation, MLflow experiment tracking, scheduled retraining with Airflow, Docker support, and a modern Streamlit dashboard.

## Features

- Local CSV dataset loading with Kaggle Telco churn support
- Missing value handling, one-hot encoding, and numerical scaling
- Feature engineering for tenure groups, average spend, service counts, contract indicators, and internet service indicators
- Logistic Regression and Random Forest model comparison
- Best model persistence with the fitted preprocessing pipeline
- MLflow run tracking for parameters, metrics, and model artifacts
- Streamlit dashboard for predictions, model performance, dataset overview, experiments, monitoring, and Airflow workflow visualization
- Docker and Docker Compose setup
- Lightweight Airflow DAG for scheduled retraining

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Apache Airflow, MLflow, Docker, Poetry, Matplotlib, and Plotly.

## Folder Structure

```text
customer-churn-pipeline/
├── app.py
├── data/
│   └── telco_churn.csv
├── models/
├── artifacts/
├── airflow/
│   └── dags/
│       └── churn_retraining_dag.py
├── backend/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
├── mlflow/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Dataset

The project includes a tiny starter CSV so the code can run immediately. For the full Kaggle dataset, download **WA_Fn-UseC_-Telco-Customer-Churn.csv** and place it anywhere under:

```text
data/
```

The loader searches recursively and prefers a Kaggle-style file matching `*Telco*Customer*Churn*.csv`. No database setup is required.

## Installation

```bash
poetry install
```

For Airflow support in the same local environment:

```bash
poetry install --extras airflow
```

## Train The Models

```bash
poetry run python backend/train.py
```

This command:

- Loads the local churn CSV
- Converts `TotalCharges` to numeric
- Creates engineered features
- Fits preprocessing with one-hot encoding and `StandardScaler`
- Trains Logistic Regression and Random Forest
- Logs MLflow runs
- Saves the best model to `models/best_model.joblib`
- Saves metrics and plots under `artifacts/`

## Run The Dashboard

```bash
poetry run streamlit run app.py
```

Open the Streamlit URL shown in the terminal. The dashboard includes:

- Home overview
- Customer churn prediction form
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, ROC curve, and feature importance
- Dataset profile and charts
- MLflow experiment tracking page
- Monitoring page for prediction volume, distribution, confidence, and last retraining time
- Airflow pipeline visualization and manual retraining button

## MLflow Tracking

MLflow stores local experiment runs in:

```text
mlflow/
```

Start the MLflow UI with:

```bash
poetry run mlflow ui --backend-store-uri mlflow --port 5000
```

Then visit `http://localhost:5000`.

## Airflow Workflow

The DAG is located at:

```text
airflow/dags/churn_retraining_dag.py
```

It automates:

1. Load data
2. Validate preprocessing
3. Train model
4. Evaluate model
5. Save model

For a simple local Airflow run, install the optional dependency first:

```bash
poetry install --extras airflow
poetry run airflow standalone
```

## Docker

Build and run the Streamlit dashboard:

```bash
docker compose up churn-dashboard
```

Run the dashboard, MLflow UI, and Airflow service:

```bash
docker compose up --build
```

Services:

- Streamlit: `http://localhost:8501`
- MLflow: `http://localhost:5000`
- Airflow: `http://localhost:8080`

## Dashboard Screenshots

Add screenshots here after running the app:

```text
docs/screenshots/home.png
docs/screenshots/prediction.png
docs/screenshots/model-performance.png
docs/screenshots/experiments.png
```

## Future Improvements

- Add XGBoost as a third candidate model
- Add automated tests for feature engineering and prediction schemas
- Add model drift checks for monthly charges and tenure
- Add CI workflow for linting and training smoke tests
- Register production model stages in MLflow Model Registry

