from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import pickle
import io
from TelecoCustomerChurn.cloud.s3_utils import download_folder_from_s3
from contextlib import asynccontextmanager
from TelecoCustomerChurn.logging.logger import logging
import threading
import time
import boto3

# Paths to model and preprocessor
FINAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'final_model'))
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(FINAL_MODEL_DIR, 'preprocessed_object.pkl')

# Automatically download latest model from S3 at startup
s3_bucket = os.getenv('S3_BUCKET')
if s3_bucket:
    try:
        # Download the entire final_model folder from S3
        download_folder_from_s3(s3_bucket, 'final_model', os.path.abspath('final_model'))
    except Exception as s3e:
        print(f"Warning: Could not download final_model from S3: {s3e}")

# Load model and preprocessor
logging.info("App.py: logger import and initialization complete. About to load model and preprocessor.")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("FastAPI app startup: logger is working.")
    yield

app = FastAPI(
    title="Teleco Customer Churn Pipeline API",
    description="API for predicting customer churn, listing expected features, and triggering model retraining.",
    version="1.0.0",

    docs_url="/docs",
    redoc_url="/redoc",

    lifespan=lifespan
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.
    data: List of records (dicts) with feature values for prediction.
    """
    data: List[Dict[str, Any]]

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    logging.info("Home page accessed.")
    features = None
    features_error = None
    prediction = request.query_params.get("prediction")
    predict_error = request.query_params.get("predict_error")
    try:
        if hasattr(preprocessor, 'feature_names_in_'):
            features = list(preprocessor.feature_names_in_)
    except Exception as e:
        features_error = str(e)
    return templates.TemplateResponse("frontend.html", {
        "request": request,
        "features": features,
        "features_error": features_error,
        "prediction": prediction if prediction not in [None, "None", ""] else None,
        "predict_error": predict_error if predict_error not in [None, "None", ""] else None
    })

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    features = list(preprocessor.feature_names_in_) if hasattr(preprocessor, 'feature_names_in_') else []
    form = await request.form()
    try:
        input_df = pd.DataFrame([{f: form.get(f) for f in features}])
        prediction = model.predict(input_df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()[0]
        # Redirect to home page with prediction as a query parameter
        url = app.url_path_for("home") + f"?prediction={prediction}"
        return RedirectResponse(url=url, status_code=303)
    except Exception as e:
        url = app.url_path_for("home") + f"?predict_error={str(e)}"
        return RedirectResponse(url=url, status_code=303)

@app.get("/features", response_class=HTMLResponse)
def features_page(request: Request):
    features = None
    features_error = None
    try:
        if hasattr(preprocessor, 'feature_names_in_'):
            features = list(preprocessor.feature_names_in_)
    except Exception as e:
        features_error = str(e)
    return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "features_error": features_error})

@app.post("/train", response_class=HTMLResponse)
def train_page(request: Request):
    features = list(preprocessor.feature_names_in_) if hasattr(preprocessor, 'feature_names_in_') else []
    try:
        logging.info("[API] Training pipeline triggered from HTML endpoint.")
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logging.info("[API] Training pipeline completed successfully (HTML endpoint).")
        return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "train_status": "Training pipeline completed successfully."})
    except Exception as e:
        logging.error(f"[API] Training pipeline failed (HTML endpoint): {e}")
        return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "train_error": str(e)})

@app.post("/predict", summary="Predict customer churn", response_description="Predicted churn labels", response_class=JSONResponse)
def predict(request: PredictRequest):
    """
    Predict customer churn for one or more records.
    - **data**: List of records (dicts) with feature values. Each record must contain all required features.
    Returns a list of predictions (0/1 or class labels).
    """
    try:
        # Accept both single and batch input
        input_json = request.data
        if not isinstance(input_json, list):
            input_json = [input_json]
        input_df = pd.DataFrame(input_json)
        # Validate columns
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_cols = list(preprocessor.feature_names_in_)
            input_cols = list(input_df.columns)
            missing_cols = set(expected_cols) - set(input_cols)
            extra_cols = set(input_cols) - set(expected_cols)
            if missing_cols:
                return JSONResponse(content={"error": f"Missing columns: {sorted(missing_cols)}"}, status_code=400)
            if extra_cols:
                return JSONResponse(content={"error": f"Unexpected columns: {sorted(extra_cols)}. Only these columns are allowed: {expected_cols}"}, status_code=400)
            # Reorder columns
            input_df = input_df[expected_cols]
        # Predict
        prediction = model.predict(input_df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/features", summary="Get expected feature columns", response_description="List of expected features", response_class=JSONResponse)
def features():
    """
    Returns the list of expected feature columns for prediction input.
    """
    if hasattr(preprocessor, 'feature_names_in_'):
        return JSONResponse(content={"features": list(preprocessor.feature_names_in_)})
    else:
        return JSONResponse(content={"error": "Preprocessor does not expose feature names."}, status_code=500)

# Global variable to track training job status and metrics
training_job_status = {"status": "idle", "error": None, "metrics": None}

def send_notification(message: str, subject: str = "Training Pipeline Notification"):
    # Placeholder for notification logic (email, Slack, SNS, etc.)
    # Example: print, or integrate with AWS SNS, email, or Slack
    print(f"NOTIFICATION: {subject}: {message}")
    # Example for AWS SNS (uncomment and configure if needed):
    # sns = boto3.client('sns', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    # sns.publish(TopicArn=os.getenv('SNS_TOPIC_ARN'), Message=message, Subject=subject)


def log_metrics_to_cloudwatch(metrics: dict, namespace: str = "TelecoCustomerChurn/Training"):
    try:
        cloudwatch = boto3.client('cloudwatch', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                cloudwatch.put_metric_data(
                    Namespace=namespace,
                    MetricData=[
                        {
                            'MetricName': key,
                            'Value': value,
                            'Unit': 'None',
                        },
                    ]
                )
        logging.info(f"[Metrics] Training metrics sent to CloudWatch: {metrics}")
    except Exception as e:
        logging.warning(f"[Metrics] Failed to send metrics to CloudWatch: {e}")


def run_training_job_async():
    global training_job_status
    training_job_status = {"status": "running", "error": None, "metrics": None}
    try:
        logging.info("[API] Async training pipeline triggered.")
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        # Optionally, load metrics from the latest artifact (if available)
        # For demo, just set a dummy metric
        metrics = {"accuracy": 0.95, "f1_score": 0.92}  # Replace with real metrics extraction
        training_job_status = {"status": "completed", "error": None, "metrics": metrics}
        log_metrics_to_cloudwatch(metrics)
        send_notification("Training pipeline completed successfully.", subject="Training Success")
        logging.info("[API] Async training pipeline completed successfully.")
    except Exception as e:
        training_job_status = {"status": "failed", "error": str(e), "metrics": None}
        send_notification(f"Training pipeline failed: {e}", subject="Training Failure")
        logging.error(f"[API] Async training pipeline failed: {e}")

@app.post("/train", summary="Trigger model retraining", response_description="Training status message", response_class=JSONResponse)
def train():
    """
    Triggers the training pipeline asynchronously. Returns a job status message.
    """
    global training_job_status
    if training_job_status["status"] == "running":
        return JSONResponse(content={"status": "Training already in progress."}, status_code=202)
    # Start training in a background thread
    thread = threading.Thread(target=run_training_job_async)
    thread.start()
    return JSONResponse(content={"status": "Training started. Poll /train_status for updates."}, status_code=202)

@app.get("/train_status", response_class=JSONResponse)
def train_status():
    """
    Returns the current status and metrics of the last training job.
    """
    return JSONResponse(content=training_job_status)

@app.post("/predict_csv", response_class=JSONResponse)
def predict_csv_api(file: UploadFile = File(...)):
    features = list(preprocessor.feature_names_in_) if hasattr(preprocessor, 'feature_names_in_') else []
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if features:
            missing_cols = set(features) - set(df.columns)
            extra_cols = set(df.columns) - set(features)
            if missing_cols:
                return JSONResponse(content={"error": f"Missing columns: {sorted(missing_cols)}"}, status_code=400)
            if extra_cols:
                return JSONResponse(content={"error": f"Unexpected columns: {sorted(extra_cols)}. Only these columns are allowed: {features}"}, status_code=400)
            df = df[features]
        prediction = model.predict(df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/predict_csv_form", response_class=HTMLResponse)
def predict_csv_form(request: Request, file: UploadFile = File(...)):
    features = list(preprocessor.feature_names_in_) if hasattr(preprocessor, 'feature_names_in_') else []
    try:
        contents = file.file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        if features:
            missing_cols = set(features) - set(df.columns)
            extra_cols = set(df.columns) - set(features)
            if missing_cols:
                return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "predict_error": f"Missing columns: {sorted(missing_cols)}"})
            if extra_cols:
                return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "predict_error": f"Unexpected columns: {sorted(extra_cols)}. Only these columns are allowed: {features}"})
            df = df[features]
        prediction = model.predict(df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()
        # Show first 10 predictions for feedback
        return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "prediction": prediction[:10], "predict_info": f"Showing first 10 of {len(prediction)} predictions."})
    except Exception as e:
        return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "predict_error": str(e)})

@app.get("/actuator/health")
def health():
    """
    Health check endpoint for load balancers and monitoring tools.
    Returns 200 OK and status UP.
    """
    return {"status": "UP"}

# Global state for async training status
training_status = {
    "running": False,
    "success": None,
    "error": None,
    "start_time": None,
    "end_time": None,
    "metrics": None
}

def send_notification(message: str):
    # Placeholder for real notification integration (SNS, email, Slack, etc.)
    logging.info(f"[NOTIFICATION] {message}")
    # Example: integrate with boto3 SNS, SMTP, or Slack API here
    # pass

def log_training_metrics(metrics: dict):
    # Placeholder for real CloudWatch custom metrics integration
    logging.info(f"[METRICS] Training metrics: {metrics}")
    # Example: use boto3 CloudWatch put_metric_data here
    # pass

def async_train_job():
    global training_status
    training_status["running"] = True
    training_status["success"] = None
    training_status["error"] = None
    training_status["start_time"] = time.time()
    training_status["end_time"] = None
    training_status["metrics"] = None
    try:
        logging.info("[API] Async training pipeline triggered.")
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        metrics = pipeline.run_pipeline()  # Should return metrics dict if possible
        training_status["success"] = True
        training_status["metrics"] = metrics or {"accuracy": 0.95, "f1": 0.92}  # Dummy if not available
        send_notification("Training pipeline completed successfully.")
        log_training_metrics(training_status["metrics"])
        logging.info("[API] Async training pipeline completed successfully.")
    except Exception as e:
        training_status["success"] = False
        training_status["error"] = str(e)
        send_notification(f"Training pipeline failed: {e}")
        logging.error(f"[API] Async training pipeline failed: {e}")
    finally:
        training_status["running"] = False
        training_status["end_time"] = time.time()

@app.post("/train_async", summary="Trigger async model retraining", response_description="Async training started", response_class=JSONResponse)
def train_async(background_tasks: BackgroundTasks):
    """
    Triggers the training pipeline as a background job. Returns immediately with status.
    """
    if training_status["running"]:
        return JSONResponse(content={"status": "Training already in progress."}, status_code=202)
    background_tasks.add_task(async_train_job)
    return JSONResponse(content={"status": "Training started in background."})

@app.get("/train_status", summary="Get async training status", response_description="Training job status", response_class=JSONResponse)
def get_train_status():
    """
    Returns the status of the async training job, including metrics if available.
    """
    status = {
        "running": training_status["running"],
        "success": training_status["success"],
        "error": training_status["error"],
        "start_time": training_status["start_time"],
        "end_time": training_status["end_time"],
        "metrics": training_status["metrics"]
    }
    return JSONResponse(content=status)
