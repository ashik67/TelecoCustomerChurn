from datetime import datetime
import os 
import sys
from TelecoCustomerChurn.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.artifact_dir = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        #self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.timestamp = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR)
        self.feature_store_file = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR)
        self.ingested_dir = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR)
        self.train_file_path = os.path.join(self.ingested_dir, training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.ingested_dir, training_pipeline.TEST_FILE_NAME)
        self.collection_name = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.train_test_split_ratio = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR)
        self.valid_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path = os.path.join(self.valid_dir, training_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path = os.path.join(self.valid_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path = os.path.join(self.invalid_dir, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.invalid_dir, training_pipeline.TEST_FILE_NAME)
        self.drift_report_dir = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)
        self.drift_report_file_path = os.path.join(self.drift_report_dir, training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)