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

## AWS & MLOps Automation

This project is fully productionized and automated for cloud deployment using modern MLOps best practices:

- **Infrastructure as Code (IaC)**: All AWS infrastructure (S3, ECR, EC2, CloudWatch, IAM, Security Groups, etc.) is provisioned and managed using Terraform scripts in the `infra/` directory.
- **Remote State Management**: Terraform state is stored remotely in S3 with state locking via DynamoDB for safe, collaborative infrastructure changes.
- **Dynamic, Secure Environment Variables**: All environment variables and secrets (S3 bucket, ECR repo, CloudWatch log group, EC2 host, etc.) are managed dynamically via Terraform outputs and injected into the CI/CD workflow. No hardcoded values.
- **CI/CD with GitHub Actions**: End-to-end automation for infrastructure provisioning, Docker build/push, and application deployment to EC2 using a single robust workflow (`.github/workflows/deploy.yml`).
- **ECR Robustness**: ECR repositories are managed with `force_delete = true` to allow safe, idempotent cleanup and redeployment.
- **Resource Import & Idempotency**: Existing AWS resources are automatically imported into Terraform state to avoid duplication and ensure idempotency.
- **Production Security**: Security group rules are defined in Terraform and can be tightened for production. IAM roles and policies are least-privilege and managed as code.
- **Resource Listing Utility**: Includes a Python script (`list_aws_resources.py`) to enumerate all major AWS resources in your account for auditing and troubleshooting.

**Key Files:**
- `infra/` — All Terraform code for AWS infrastructure
- `.github/workflows/deploy.yml` — Main CI/CD workflow (infra + app deploy)
- `list_aws_resources.py` — Script to list all AWS resources

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

## ETL Pipeline

This project uses a modular ETL pipeline to automate data ingestion, transformation, drift detection, and (optionally) retraining. Below is a summary of each step and how it is implemented:

### 1. Data Fetch & Ingestion
- Loads raw data from `Data/TelecoCustomerChurn.csv` (or external source if configured).
- Validates schema using `data_schema/schema.yaml`.
- Stores ingested data in the `artifacts/<timestamp>/data_ingestion/` directory for reproducibility.
- Code: `TelecoCustomerChurn/components/data_ingestion.py`

### 2. Data Validation
- Checks for missing values, schema mismatches, and data integrity.
- Logs validation results and stores validated data in `artifacts/<timestamp>/data_validation/`.
- Code: `TelecoCustomerChurn/components/data_validation.py`

### 3. Data Transformation
- Applies feature engineering, encoding, and scaling using a preprocessor pipeline.
- Saves transformed data and preprocessor object in `artifacts/<timestamp>/data_transformation/`.
- Code: `TelecoCustomerChurn/components/data_transformation.py`

### 4. Model Training
- Trains the model on transformed data using the most important features (see above).
- Evaluates model performance and logs metrics to MLflow/Dagshub.
- Saves model and artifacts in `final_model/` and `artifacts/<timestamp>/model_training/`.
- Code: `TelecoCustomerChurn/components/model_trainer.py`

### 5. Drift Detection & Retraining Logic
- After data validation, a drift report is generated and checked for significant distribution changes between new and training data.
- If drift is detected (`overall_drift_found: true` in the drift report), the pipeline proceeds with retraining the model.
- If no drift is detected and a model already exists, retraining is skipped for efficiency.
- If no model exists (first run or missing), training is always performed.
- Drift detection and retraining logic are implemented in `TelecoCustomerChurn/pipeline/training_pipeline.py`.

---

## Pipeline Step Implementation Details

Each stage of the ML pipeline is modularized and productionized for reliability, reproducibility, and easy orchestration:

- **Data Ingestion**: Raw data is loaded from CSV or other sources, validated against a schema, and stored in the `artifacts/` directory with timestamped runs for traceability.
- **Data Validation**: The pipeline checks for missing values, schema mismatches, and data drift. Validation reports are generated and stored as artifacts.
- **Data Transformation**: Features are preprocessed (encoding, scaling, imputation) using robust pipelines. The preprocessor object is saved alongside the model for consistent inference.
- **Model Training**: The pipeline trains a classifier (e.g., RandomForest, XGBoost) using the transformed data. Hyperparameters and metrics are tracked with MLflow and/or Dagshub.
- **Model Evaluation**: The trained model is evaluated on a holdout set. Metrics (accuracy, ROC-AUC, etc.) and feature importances are logged for experiment tracking.
- **Artifact Management**: All models, preprocessors, and reports are versioned and saved in `final_model/` and `artifacts/` for reproducibility and rollback.
- **Experiment Tracking**: MLflow and Dagshub are used to log parameters, metrics, and artifacts for each run, enabling experiment comparison and auditability.
- **Serving & Inference**: The latest model and preprocessor are loaded by the FastAPI app for real-time and batch predictions. The API ensures input validation and returns clear results.
- **Retraining & Orchestration**: The `/train` endpoint or your own orchestration logic can trigger retraining. The pipeline supports ETL, drift detection, and conditional retraining, making it robust and production-ready.

---

## Environment Variables & Secrets

This project uses environment variables to securely manage sensitive information and configuration. The following variables are required:

| Variable     | Description                                 | Where to Set                |
|--------------|---------------------------------------------|-----------------------------|
| `MONGO_URI`  | MongoDB connection string (sensitive)       | `.env` (local), GitHub Secret (CI/CD) |
| `S3_BUCKET`  | Name of the S3 bucket for model artifacts   | `.env` (local), GitHub Variable/Secret (CI/CD) |

### Local Development
- Create a `.env` file in the project root:
  ```ini
  MONGO_URI="your-mongodb-uri"
  S3_BUCKET="your-s3-bucket-name"
  ```
- **Do not commit `.env` to git** (already in `.gitignore`).
- The app will automatically load these variables if you use `python-dotenv` or similar, or you can set them in your shell.

### Production / CI/CD (GitHub Actions)
- Store `MONGO_URI` as a **GitHub Actions Secret**.
- Store `S3_BUCKET` as a **GitHub Actions Variable** or Secret.
- The deployment workflow injects these variables into the Docker container at runtime using the `-e` flag.
- No sensitive values are ever stored in the repository.

### Docker
- When running locally with Docker, pass environment variables using `--env-file` or `-e`:
  ```sh
  docker run --env-file .env -p 8000:8000 teleco-churn-fastapi
  ```
  or
  ```sh
  docker run -e MONGO_URI="your-mongodb-uri" -e S3_BUCKET="your-s3-bucket-name" -p 8000:8000 teleco-churn-fastapi
  ```

---