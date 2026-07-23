# Customer Churn Prediction Pipeline

An end-to-end machine learning project for predicting telecom customer churn. The project covers the complete ML workflow, including data preprocessing, feature engineering, model training, experiment tracking with MLflow, workflow automation using Airflow, containerization with Docker, and an interactive Streamlit dashboard.

## Overview

This project demonstrates how a machine learning model can be developed and managed beyond a notebook. It includes:

- Data preprocessing and feature engineering
- Training and evaluation of multiple machine learning models
- Experiment tracking using MLflow
- Automated retraining with Apache Airflow
- Interactive visualization and prediction using Streamlit
- Docker support for reproducible deployment

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- MLflow
- Apache Airflow
- Docker
- Poetry
- Plotly
- Matplotlib

## Project Structure

```text
customer-churn-pipeline/
│
├── app.py                     # Streamlit application
├── backend/
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── airflow/
│   └── dags/
│       └── churn_retraining_dag.py
│
├── data/
├── models/
├── artifacts/
├── mlflow/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Dataset

## Dataset

This project uses the **Telco Customer Churn** dataset.

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn

After downloading, place the file

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

inside the `data/` directory as Telco-Churn.csv:

```text
data/
└── Telco-Churn.csv
```

The dataset is not included in this repository due to licensing and repository size considerations. :contentReference[oaicite:0]{index=0}


## Installation

Clone the repository:

```bash
git clone https://github.com/Rishika297/customer-churn-pipeline.git

cd customer-churn-pipeline
```

Install project dependencies:

```bash
poetry install
```

Install Airflow dependencies (optional):

```bash
poetry install --extras airflow
```

## Training

Train the machine learning models:

```bash
poetry run python backend/train.py
```

The training pipeline performs the following steps:

- Load the dataset
- Apply preprocessing and feature engineering
- Train Logistic Regression and Random Forest models
- Compare model performance
- Log experiments with MLflow
- Save the best-performing model

## Running the Dashboard

Launch the Streamlit application:

```bash
poetry run streamlit run app.py
```

The dashboard includes:

- Project overview
- Customer churn prediction
- Model performance metrics
- Dataset exploration
- MLflow experiment tracking
- Monitoring dashboard
- Airflow workflow overview

## MLflow

Start the MLflow tracking server:

```bash
poetry run mlflow ui --backend-store-uri mlflow --port 5000
```

Access the interface at:

```
http://localhost:5000
```

## Airflow

The retraining workflow is located at:

```text
airflow/dags/churn_retraining_dag.py
```

Pipeline:

```text
Load Data
    ↓
Preprocess
    ↓
Train
    ↓
Evaluate
    ↓
Save Model
```

Run Airflow locally:

```bash
poetry install --extras airflow

poetry run airflow standalone
```

## Docker

Run only the Streamlit application:

```bash
docker compose up churn-dashboard
```

Run all services:

```bash
docker compose up --build
```

| Service | URL |
|----------|-----|
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

## Screenshots

Add screenshots after running the project.

- Home page
- Prediction page
- Model performance
- MLflow experiments
- Airflow workflow

## Future Work

Potential improvements include:

- Adding XGBoost and LightGBM models
- Hyperparameter tuning
- Model drift monitoring
- Unit testing
- CI/CD using GitHub Actions
- MLflow Model Registry

## Motivation

The goal of this project was to build a complete machine learning pipeline rather than a notebook-based model. It demonstrates how data preprocessing, model training, experiment tracking, workflow automation, and deployment can be integrated into a single reproducible application.
