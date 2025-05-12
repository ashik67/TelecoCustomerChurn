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