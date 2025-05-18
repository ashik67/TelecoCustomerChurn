from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import pickle
import io
from TelecoCustomerChurn.cloud.s3_utils import download_folder_from_s3
from TelecoCustomerChurn.logging import logger
import logging

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
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

app = FastAPI(
    title="Teleco Customer Churn Pipeline API",
    description="API for predicting customer churn, listing expected features, and triggering model retraining.",
    version="1.0.0",

    docs_url="/docs",
    redoc_url="/redoc"
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
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return templates.TemplateResponse("frontend.html", {"request": request, "features": features, "train_status": "Training pipeline completed successfully."})
    except Exception as e:
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

@app.post("/train", summary="Trigger model retraining", response_description="Training status message", response_class=JSONResponse)
def train():
    """
    Triggers the training pipeline. Returns a success or error message.
    """
    try:
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return JSONResponse(content={"status": "Training pipeline completed successfully."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

@app.on_event("startup")
def startup_event():
    logger.info("FastAPI app startup: logger is working.")
