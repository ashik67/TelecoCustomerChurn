import os
import sys
import numpy as np
import pandas as pd

"""This module contains constants and configuration settings
 for the training pipeline of the Telecom Customer Churn project."""

# Name of the target column in the dataset
TARGET_COLUMN_NAME = "Churn"
# Name of the ML pipeline
PIPELINE_NAME = "TelecomCustomerChurn"
# Root directory for all pipeline artifacts
ARTIFACT_DIR = "artifacts"
# Main dataset filename
FILENAME = "TelecoCustomerChurn.csv"

# Filenames for train and test splits (CSV format)
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

# Filenames for transformed numpy arrays (features and targets)
TRANSFORMED_TRAIN_FILE_NAME = "train.npy"  # Transformed training features (numpy array)
TRANSFORMED_TEST_FILE_NAME = "test.npy"    # Transformed test features (numpy array)
TRANSFORMED_TRAIN_TARGET_FILE_NAME = "train_target.npy"  # Encoded training target (numpy array)
TRANSFORMED_TEST_TARGET_FILE_NAME = "test_target.npy"    # Encoded test target (numpy array)

# Path to the schema file for data validation
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""This module contains constants and configuration settings
 for the data ingestion process in the Telecom Customer Churn project."""

# MongoDB collection and database names
DATA_INGESTION_COLLECTION_NAME = "TelecomCustomerChurn"
DATA_INGESTION_DATABASE_NAME = "TelecomCustomerChurn"
# Directory names for data ingestion artifacts
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
# Ratio for splitting data into train and test
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Directory names for data validation artifacts
DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_VALID_DIR = "valid"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.yaml"

# Directory names for data transformation artifacts
DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DIR = "preprocessed_object"

# Imputer params for numeric columns (use KNNImputer)
DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 5,  # Number of neighbors for KNN imputation
    "weights": "uniform"
}

# Imputer params for categorical columns (use SimpleImputer with most frequent strategy)
DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "strategy": "most_frequent",
    "fill_value": None,
    "add_indicator": False
}

"""
This module contains constants and configuration settings
 for the model training process in the Telecom Customer Churn project.
"""
# Directory names for model training artifacts
MODEL_TRAINING_DIR = "model_training"  # Root directory for model training artifacts
MODEL_TRAINING_MODEL_DIR = "model"      # Directory for storing the trained model
MODEL_TRAINING_METRICS_DIR = "metrics"  # Directory for storing model metrics
MODEL_TRAINING_REPORT_DIR = "report"    # Directory for storing model training report
# Filenames for model training outputs
MODEL_TRAINING_MODEL_FILE_NAME = "model.pkl"         # Filename for the trained model
MODEL_TRAINING_METRICS_FILE_NAME = "metrics.yaml"    # Filename for model metrics
MODEL_TRAINING_REPORT_FILE_NAME = "report.yaml"      # Filename for model training report
# Model training expectations and thresholds
MODEL_TRAINING_EXPECTED_SCORE = 0.7  # Expected accuracy score for the model
MODEL_TRAINING_FITTING_THRESHOLDS = {
    "overfitting": 0.05,  # Threshold for overfitting detection
    "underfitting": 0.05  # Threshold for underfitting detection
}
