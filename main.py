from TelecoCustomerChurn.components.data_ingestion import DataIngestion
from TelecoCustomerChurn.components.data_validation import DataValidation
from TelecoCustomerChurn.components.data_transformation import DataTransformation
from TelecoCustomerChurn.components.model_trainer import ModelTrainer
from TelecoCustomerChurn.components.model_pusher import ModelPusher
from TelecoCustomerChurn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, ModelPusherArtifact
from TelecoCustomerChurn.entity.config_entity import DataIngestionConfig
from TelecoCustomerChurn.entity.config_entity import DataValidationConfig
from TelecoCustomerChurn.entity.config_entity import DataTransformationConfig
from TelecoCustomerChurn.entity.config_entity import ModelTrainingConfig
from TelecoCustomerChurn.entity.config_entity import ModelPusherConfig
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        # Initialize the training pipeline config
        training_pipeline_config = TrainingPipelineConfig()
        
        # Initialize the data ingestion config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        
        # Initialize the data ingestion component
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        
        # Start the data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

        # Initialize the data validation config
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)

        # Initialize the data validation component
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)

        # Start the data validation process
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation artifact: {data_validation_artifact}")

        # Initialize the data transformation config
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        # Initialize the data transformation component
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_validation_artifact=data_validation_artifact)
        # Start the data transformation process
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data transformation artifact: {data_transformation_artifact}")
        
        Model_Training_Config = ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        # Initialize the model training component
        Model_Trainer = ModelTrainer(model_training_config=Model_Training_Config, data_transformation_artifact=data_transformation_artifact) 
        # Start the model training process
        model_trainer_artifact = Model_Trainer.initiate_model_training()
        logging.info(f"Model training artifact: {model_trainer_artifact}")

        # --- Model Pusher Integration ---
        model_pusher_config = ModelPusherConfig(
            artifact_dir=training_pipeline_config.artifact_dir
        )
        model_pusher = ModelPusher(
            model_pusher_config=model_pusher_config,
            model_training_artifact=model_trainer_artifact,
            data_transformation_artifact=data_transformation_artifact
        )
        model_pusher_artifact = model_pusher.push_model()
        logging.info(f"Model pusher artifact: {model_pusher_artifact}")
        
    except Exception as e:
        raise CustomerChurnException(e, sys) from e