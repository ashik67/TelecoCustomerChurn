from TelecoCustomerChurn.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainingConfig, ModelPusherConfig
from TelecoCustomerChurn.components.data_ingestion import DataIngestion
from TelecoCustomerChurn.components.data_validation import DataValidation
from TelecoCustomerChurn.components.data_transformation import DataTransformation
from TelecoCustomerChurn.components.model_trainer import ModelTrainer
from TelecoCustomerChurn.components.model_pusher import ModelPusher
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
import sys

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def run_pipeline(self):
        try:
            # Data Ingestion
            data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            # Data Validation
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            # Fail-fast if validation fails
            if not getattr(data_validation_artifact, 'validation_status', True):
                logging.error("Data validation failed. Stopping pipeline.")
                raise CustomerChurnException("Data validation failed. See logs for details.", sys)

            # Data Transformation
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")

            # Model Training
            model_training_config = ModelTrainingConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_training_config=model_training_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_training()
            logging.info(f"Model training artifact: {model_trainer_artifact}")

            # Model Pusher
            model_pusher_config = ModelPusherConfig(artifact_dir=self.training_pipeline_config.artifact_dir)
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config, model_training_artifact=model_trainer_artifact, data_transformation_artifact=data_transformation_artifact)
            model_pusher_artifact = model_pusher.push_model()
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")

        except Exception as e:
            logging.error(f"Exception in TrainingPipeline: {e}")
            raise CustomerChurnException(e, sys) from e
