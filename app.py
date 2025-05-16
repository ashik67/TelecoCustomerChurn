from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import os
import pickle

# Paths to model and preprocessor
FINAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'final_model'))
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(FINAL_MODEL_DIR, 'preprocessed_object.pkl')

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

class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.
    data: List of records (dicts) with feature values for prediction.
    """
    data: List[Dict[str, Any]]

@app.post("/predict", summary="Predict customer churn", response_description="Predicted churn labels")
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
                return {"error": f"Missing columns: {sorted(missing_cols)}"}
            if extra_cols:
                return {"error": f"Unexpected columns: {sorted(extra_cols)}. Only these columns are allowed: {expected_cols}"}
            # Reorder columns
            input_df = input_df[expected_cols]
        # Predict
        prediction = model.predict(input_df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.get("/features", summary="Get expected feature columns", response_description="List of expected features")
def features():
    """
    Returns the list of expected feature columns for prediction input.
    """
    if hasattr(preprocessor, 'feature_names_in_'):
        return {"features": list(preprocessor.feature_names_in_)}
    else:
        return {"error": "Preprocessor does not expose feature names."}

@app.post("/train", summary="Trigger model retraining", response_description="Training status message")
def train():
    """
    Triggers the training pipeline. Returns a success or error message.
    """
    try:
        from TelecoCustomerChurn.pipeline.training_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return {"status": "Training pipeline completed successfully."}
    except Exception as e:
        return {"error": str(e)}
