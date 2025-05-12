from TelecoCustomerChurn.components.data_ingestion import DataIngestion
from TelecoCustomerChurn.components.data_validation import DataValidation
from TelecoCustomerChurn.entity.config_entity import DataIngestionConfig
from TelecoCustomerChurn.entity.config_entity import DataValidationConfig
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


        
    except Exception as e:
        raise CustomerChurnException(e, sys) from e