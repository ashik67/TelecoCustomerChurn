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


"""This module contains constants and configuration settings
 for the data ingestion process in the Telecom Customer Churn project."""

DATA_INGESTION_COLLECTION_NAME = "TelecomCustomerChurn"
DATA_INGESTION_DATABASE_NAME = "TelecomCustomerChurn"
DATA_INGESTION_DIR = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2