# 🚀 Customer Churn Prediction Pipeline

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. This project goes beyond just training a model—it includes feature engineering, experiment tracking, workflow automation, Docker support, and an interactive dashboard to explore predictions and model performance.

---

## ✨ What this project does

- 📊 Predicts customer churn using machine learning
- 🧹 Cleans and preprocesses raw telecom customer data
- ⚙️ Engineers additional features to improve model performance
- 🤖 Trains and compares multiple ML models
- 📈 Tracks experiments with MLflow
- 🔄 Automates retraining using Airflow
- 🖥️ Provides an interactive Streamlit dashboard
- 🐳 Runs easily with Docker and Docker Compose

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- Streamlit
- MLflow
- Apache Airflow
- Docker
- Poetry
- Plotly & Matplotlib

---

## 📂 Project Structure

```text
customer-churn-pipeline/
│
├── app.py                     # Streamlit dashboard
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

---

## 📊 Dataset

The project works with the **Telco Customer Churn** dataset.

A small sample dataset is included so the project runs out of the box.

For the complete dataset, download:

**WA_Fn-UseC_-Telco-Customer-Churn.csv**

and place it inside:

```text
data/
```

No database setup required.

---

## ⚡ Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/customer-churn-pipeline.git

cd customer-churn-pipeline
```

Install dependencies

```bash
poetry install
```

(Optional) Install Airflow support

```bash
poetry install --extras airflow
```

---

## 🏋️ Train the Model

```bash
poetry run python backend/train.py
```

This pipeline will:

- Load the dataset
- Perform feature engineering
- Preprocess the data
- Train Logistic Regression and Random Forest
- Compare model performance
- Track experiments with MLflow
- Save the best-performing model

---

## 🎯 Run the Dashboard

```bash
poetry run streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

The dashboard includes:

- 🏠 Project Overview
- 🔮 Customer Churn Prediction
- 📈 Model Performance
- 📊 Dataset Insights
- 🧪 MLflow Experiments
- 📉 Monitoring Dashboard
- 🔄 Airflow Workflow

---

## 📈 MLflow

Start the MLflow UI

```bash
poetry run mlflow ui --backend-store-uri mlflow --port 5000
```

Visit

```
http://localhost:5000
```

to explore experiment history.

---

## 🔄 Airflow

The retraining DAG lives here:

```text
airflow/dags/churn_retraining_dag.py
```

Workflow:

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

---

## 🐳 Docker

Run only the dashboard

```bash
docker compose up churn-dashboard
```

Run everything

```bash
docker compose up --build
```

Available services

| Service | URL |
|---------|-----|
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

---

## 📸 Screenshots

> Add screenshots after running the project.

- Home
- Prediction Page
- Model Performance
- MLflow Experiments
- Airflow Workflow

---

## 🚀 Future Improvements

- Add XGBoost and LightGBM
- Hyperparameter tuning
- Model drift monitoring
- Unit tests
- CI/CD with GitHub Actions
- MLflow Model Registry

---

## 💡 Why I built this

I wanted to build a project that demonstrates the complete machine learning workflow—from raw data to deployment—rather than just training a model in a notebook. This project helped me explore experiment tracking with MLflow, workflow automation with Airflow, containerization using Docker, and building an interactive dashboard with Streamlit.