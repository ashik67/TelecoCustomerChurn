import bentoml
from bentoml.io import JSON
import pandas as pd
import os
import pickle

# Define paths to the final model and preprocessor
FINAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'final_model'))
MODEL_PATH = os.path.join(FINAL_MODEL_DIR, 'model.pkl')
PREPROCESSOR_PATH = os.path.join(FINAL_MODEL_DIR, 'preprocessed_object.pkl')

# Load the model and preprocessor using pickle
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)

svc = bentoml.Service("teleco_churn_classifier")

@svc.api(input=JSON(), output=JSON(), route="/predict")
async def predict(input_json):
    """
    Teleco Customer Churn Prediction API endpoint.
    Accepts JSON input, preprocesses it, and returns model predictions.
    Handles both single and batch input, and ensures correct feature order.
    """
    try:
        # Handle both single dict and list of dicts
        if isinstance(input_json, dict):
            input_df = pd.DataFrame([input_json])
        elif isinstance(input_json, list):
            input_df = pd.DataFrame(input_json)
        else:
            return {"error": "Input must be a JSON object or list of objects."}

        # Ensure columns match what the preprocessor expects (if available)
        if hasattr(preprocessor, 'feature_names_in_'):
            expected_cols = list(preprocessor.feature_names_in_)
            # For batch input, check every dict for exact keys and order
            if isinstance(input_json, list):
                for idx, row in enumerate(input_json):
                    row_keys = list(row.keys())
                    if row_keys != expected_cols:
                        return {"error": f"At index {idx}, input keys do not match expected columns or order.\nExpected: {expected_cols}\nReceived: {row_keys}"}
            input_cols = list(input_df.columns)
            input_cols_set = set(input_cols)
            expected_cols_set = set(expected_cols)
            missing_cols = expected_cols_set - input_cols_set
            extra_cols = input_cols_set - expected_cols_set
            if missing_cols:
                return {"error": f"Missing columns: {sorted(missing_cols)}"}
            if extra_cols:
                return {"error": f"Unexpected columns: {sorted(extra_cols)}. Only these columns are allowed: {expected_cols}"}
            # Check for exact column match and order
            if input_cols != expected_cols:
                return {"error": f"Input columns must exactly match and be in this order: {expected_cols}. Received: {input_cols}"}
            # Check for empty input
            if input_df.shape[0] == 0:
                return {"error": "Input must contain at least one record."}
            # Always select only the expected columns, in order
            input_df = input_df[expected_cols]
            # Debug: print shape and columns
            print(f"[DEBUG] input_df.shape: {input_df.shape}")
            print(f"[DEBUG] input_df.columns: {list(input_df.columns)}")
            print(f"[DEBUG] expected_cols: {expected_cols}")

        # Pass input_df directly to model.predict, which handles transformation
        prediction = model.predict(input_df)
        if hasattr(prediction, 'tolist'):
            prediction = prediction.tolist()
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@svc.api(input=JSON(), output=JSON(), route="/features")
async def features(_):
    """
    Returns the list of expected feature columns for prediction input.
    """
    if hasattr(preprocessor, 'feature_names_in_'):
        return {"features": list(preprocessor.feature_names_in_)}
    else:
        return {"error": "Preprocessor does not expose feature names."}

@svc.api(input=JSON(), output=JSON(), route="/train")
async def train(_):
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
