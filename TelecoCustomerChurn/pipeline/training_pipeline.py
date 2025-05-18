from TelecoCustomerChurn.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainingConfig, ModelPusherConfig
from TelecoCustomerChurn.components.data_ingestion import DataIngestion
from TelecoCustomerChurn.components.data_validation import DataValidation
from TelecoCustomerChurn.components.data_transformation import DataTransformation
from TelecoCustomerChurn.components.model_trainer import ModelTrainer
from TelecoCustomerChurn.components.model_pusher import ModelPusher
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.cloud.s3_utils import download_folder_from_s3
import boto3
import sys
import os

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        logging.info("Initialized TrainingPipeline with config: %s", self.training_pipeline_config)

    def model_exists(self):
        """
        Check if the model exists in S3 (preferred) or locally in final_model/.
        Returns True if model exists, False otherwise.
        """
        s3_bucket = os.getenv('S3_BUCKET')
        s3_prefix = 'final_model/'
        logging.info(f"Checking for model existence in S3 bucket: {s3_bucket}, prefix: {s3_prefix}")
        if s3_bucket:
            try:
                s3 = boto3.client('s3')
                response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
                if 'Contents' in response and any(obj['Key'].endswith('model.pkl') for obj in response['Contents']):
                    logging.info("Model found in S3 at s3://%s/%s", s3_bucket, s3_prefix)
                    return True
                else:
                    logging.info("Model not found in S3 at s3://%s/%s", s3_bucket, s3_prefix)
            except Exception as e:
                logging.warning(f"S3 model existence check failed: {e}")
        # Fallback: check local final_model/
        local_model_path = os.path.join(os.getcwd(), 'final_model', 'model.pkl')
        logging.info(f"Checking for local model at: {local_model_path}")
        exists = os.path.exists(local_model_path)
        if exists:
            logging.info("Model found locally at %s", local_model_path)
        else:
            logging.info("Model not found locally at %s", local_model_path)
        return exists

    def run_pipeline(self):
        try:
            logging.info("Starting pipeline run...")
            # Data Ingestion
            logging.info("Starting data ingestion...")
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            # Data Validation
            logging.info("Starting data validation...")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            # Load drift report
            drift_report_path = getattr(data_validation_artifact, 'drift_report_file_path', None)
            drift_found = False
            if drift_report_path and os.path.exists(drift_report_path):
                import yaml
                logging.info(f"Loading drift report from: {drift_report_path}")
                with open(drift_report_path, 'r') as f:
                    drift_report = yaml.safe_load(f)
                drift_found = drift_report.get('overall_drift_found', False)
                logging.info(f"Drift report loaded. Drift found: {drift_found}")
            else:
                logging.warning(f"Drift report not found at: {drift_report_path}")

            # Conditional retraining logic
            model_exists = self.model_exists()
            if not model_exists:
                logging.info("No existing model found (first run or model missing). Proceeding with training.")
            elif drift_found:
                logging.info("Data drift detected. Proceeding with retraining.")
            else:
                logging.info("No drift detected and model exists. Skipping retraining.")
                return  # Exit pipeline early

            # Data Transformation
            logging.info("Starting data transformation...")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            # Model Training
            logging.info("Starting model training...")
            model_training_config = ModelTrainingConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_training_config=model_training_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info(f"Model training artifact: {model_trainer_artifact}")

            # Model Pusher
            logging.info("Starting model pusher...")
            model_pusher_config = ModelPusherConfig(artifact_dir=self.training_pipeline_config.artifact_dir)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config, model_training_artifact=model_trainer_artifact, data_transformation_artifact=data_transformation_artifact)
            model_pusher_artifact = model_pusher.push_model()
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            logging.info("Pipeline run completed successfully.")

        except Exception as e:
            logging.error(f"Exception in TrainingPipeline: {e}")
            raise CustomerChurnException(e, sys) from e
