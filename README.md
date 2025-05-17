# TelecoCustomerChurn ML Pipeline & API

This project provides a robust, production-ready machine learning pipeline and API for predicting customer churn in the telecommunications industry. It includes data ingestion, validation, transformation, model training, experiment tracking, artifact management, and real-time model serving with a modern web frontend and REST API.

## Dataset Overview

The TelecoCustomerChurn dataset is designed to analyze customer behavior and predict churn in the telecommunications industry. It contains information about customer demographics, account details, services subscribed, and usage patterns. The dataset is commonly used for machine learning tasks such as classification and predictive modeling.

### Key Features:
- **Customer Demographics**: Gender, age, and other personal details.
- **Account Information**: Contract type, tenure, and payment methods.
- **Services Subscribed**: Internet, phone, and additional services like streaming.
- **Usage Patterns**: Monthly charges, total charges, and usage metrics.
- **Churn Label**: Indicates whether the customer has churned (Yes/No).

This dataset is ideal for building models to identify factors influencing customer churn and implementing strategies to improve customer retention.

---

## Features

- **End-to-End ML Pipeline**: Data ingestion, validation, transformation, model training, evaluation, and artifact management.
- **Experiment Tracking**: MLflow and Dagshub integration for experiment and artifact tracking.
- **Robust Model Packaging**: Model and preprocessor are saved together for consistent inference.
- **FastAPI Service**: Real-time REST API for prediction, retraining, and feature introspection.
- **Jinja2 Web Frontend**: User-friendly web interface for single/batch prediction and retraining.
- **Batch Prediction via CSV**: Upload CSV files for large-scale predictions.
- **OpenAPI Docs**: Interactive API documentation at `/docs` and `/redoc`.
- **Dockerized**: Production-ready Dockerfile for easy deployment.

---

## API Endpoints

- `GET /` : Web frontend for interactive prediction and retraining.
- `POST /predict` : Predict churn for one or more records (JSON body).
- `POST /predict_csv` : Upload a CSV file for batch prediction.
- `GET /features` : Get the list of expected input features.
- `POST /train` : Trigger model retraining pipeline.
- `GET /docs` : Swagger UI (OpenAPI docs).
- `GET /redoc` : ReDoc (OpenAPI docs).

---

## Usage

### 1. Install dependencies
```sh
pip install -r requirements.txt
```

### 2. Run the FastAPI app
```sh
uvicorn app:app --reload
```
Visit [http://localhost:8000/](http://localhost:8000/) for the web UI.

### 3. Docker (Production)
```sh
docker build -t teleco-churn-fastapi .
docker run -p 8000:8000 teleco-churn-fastapi
```

### 4. API Example (JSON)
```json
POST /predict
{
  "data": [
    {
      "customerID": "12345-ABCDE",
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    }
  ]
}
```

---

## Important Features Used for Prediction

The model is trained to predict churn using the following most important features:

- **SeniorCitizen**
- **tenure**
- **InternetService**
- **Contract**
- **PaymentMethod**
- **MonthlyCharges**
- **TotalCharges**

Other features may be present in the dataset, but these are the primary drivers for the model's predictions.

---

## Project Structure

- `app.py` : FastAPI app with all endpoints and Jinja2 integration
- `templates/frontend.html` : Web UI template
- `final_model/` : Latest model and preprocessor artifacts
- `TelecoCustomerChurn/` : Pipeline, components, and utilities
- `requirements.txt` : All dependencies
- `Dockerfile` : Production containerization
- `README.md` : Project documentation

---