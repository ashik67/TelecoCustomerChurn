import os
import sys
import numpy as np
import pandas as pd

"""This module contains constants and configuration settings
 for the training pipeline of the Telecom Customer Churn project."""

TARGET_COLUMN_NAME = "Churn"
PIPELINE_NAME = "TelecomCustomerChurn"
ARTIFACT_DIR = "artifacts"
FILENAME = "TelecoCustomerChurn.csv"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"


TRANSFORMED_TRAIN_FILE_NAME = "train.npy"
TRANSFORMED_TEST_FILE_NAME = "test.npy"
TRANSFORMED_TRAIN_TARGET_FILE_NAME = "train_target.npy"
TRANSFORMED_TEST_TARGET_FILE_NAME = "test_target.npy"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")


"""This module contains constants and configuration settings
 for the data ingestion process in the Telecom Customer Churn project."""

DATA_INGESTION_COLLECTION_NAME = "TelecomCustomerChurn"
DATA_INGESTION_DATABASE_NAME = "TelecomCustomerChurn"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

DATA_VALIDATION_DIR = "data_validation"
DATA_VALIDATION_VALID_DIR = "valid"
DATA_VALIDATION_INVALID_DIR = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.yaml"


DATA_TRANSFORMATION_DIR = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"
DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DIR = "preprocessed_object"

# Imputer params for numeric columns (use KNN)
DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 5,  # You can adjust this as needed
    "weights": "uniform"
}

# Imputer params for categorical columns (use most frequent)
DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "strategy": "most_frequent",
    "fill_value": None,
    "add_indicator": False
}