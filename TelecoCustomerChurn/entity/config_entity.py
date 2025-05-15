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

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR)
        self.transformed_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DIR)
        self.preprocessed_object_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_PREPROCESSED_OBJECT_DIR)
        self.transformed_train_file_path = os.path.join(self.transformed_dir, training_pipeline.TRANSFORMED_TRAIN_FILE_NAME)
        self.transformed_test_file_path = os.path.join(self.transformed_dir, training_pipeline.TRANSFORMED_TEST_FILE_NAME)
        self.transformed_train_target_file_path = os.path.join(self.transformed_dir, training_pipeline.TRANSFORMED_TRAIN_TARGET_FILE_NAME)
        self.transformed_test_target_file_path = os.path.join(self.transformed_dir, training_pipeline.TRANSFORMED_TEST_TARGET_FILE_NAME)
        self.preprocessed_object_file_path = os.path.join(self.preprocessed_object_dir, "preprocessed_object.pkl")
        self.target_column_name = training_pipeline.TARGET_COLUMN_NAME
        self.numeric_imputer_params = training_pipeline.DATA_TRANSFORMATION_NUMERIC_IMPUTER_PARAMS
        self.categorical_imputer_params = training_pipeline.DATA_TRANSFORMATION_CATEGORICAL_IMPUTER_PARAMS


class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_training_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINING_DIR)
        self.model_file_path = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_MODEL_FILE_NAME)
        self.metrics_file_path = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_METRICS_FILE_NAME)
        self.report_file_path = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_REPORT_FILE_NAME)
        self.model_dir = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_MODEL_DIR)
        self.metrics_dir = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_METRICS_DIR)
        self.report_dir = os.path.join(self.model_training_dir, training_pipeline.MODEL_TRAINING_REPORT_DIR)
        self.expected_score = training_pipeline.MODEL_TRAINING_EXPECTED_SCORE
        self.fitting_thresholds = training_pipeline.MODEL_TRAINING_FITTING_THRESHOLDS
        self.training_timestamp = training_pipeline_config.timestamp

class ModelPusherConfig:
    def __init__(self, artifact_dir: str, final_model_dir: str = None):
        self.artifact_dir = artifact_dir
        # Default to 'final_model' at project root if not specified
        self.final_model_dir = final_model_dir or os.path.abspath(os.path.join(os.getcwd(), 'final_model'))
