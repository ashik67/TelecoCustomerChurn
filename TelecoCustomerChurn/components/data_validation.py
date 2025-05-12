from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.entity.config_entity import DataValidationConfig
from TelecoCustomerChurn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from TelecoCustomerChurn.constants.training_pipeline import SCHEMA_FILE_PATH
from TelecoCustomerChurn.utils.main_utils import read_yaml_file, save_yaml_file
import os
import sys
import pandas as pd
from typing import List
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def validate_schema(self, data: pd.DataFrame, schema: dict) -> None:
        """
        Validate the schema of the DataFrame against the provided schema.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            schema (dict): The schema to validate against.

        Raises:
            ValueError: If the DataFrame does not match the schema.
        """
        try:
            for column, properties in schema.items():
                if column.length != schema[column]['length']:
                    raise ValueError(f"Column '{column}' has incorrect length. Expected {schema[column]['length']}, got {len(data[column])}.")
                if column not in data.columns:
                    raise ValueError(f"Column '{column}' is missing from the DataFrame.")
                if data[column].dtype != properties['type']:
                    raise ValueError(f"Column '{column}' has incorrect type. Expected {properties['type']}, got {data[column].dtype}.")
                if 'min' in properties and data[column].min() < properties['min']:
                    raise ValueError(f"Column '{column}' has values less than minimum allowed value {properties['min']}.")
                if 'max' in properties and data[column].max() > properties['max']:
                    raise ValueError(f"Column '{column}' has values greater than maximum allowed value {properties['max']}.")
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
    
    def check_data_drift(self, train_data: pd.DataFrame, test_data: pd.DataFrame, schema: dict) -> dict:
        """
        Check for data drift between the train and test datasets.

        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.
            schema (dict): The schema to validate against.

        Returns:
            dict: A report of the data drift analysis.
        """
        try:
            drift_report = {}
            drift_found = False
            threshold = 0.05  # p-value threshold for drift
            for column in schema.keys():
                if column in train_data.columns and column in test_data.columns:
                    ks_statistic, p_value = ks_2samp(train_data[column], test_data[column])
                    drift = p_value < threshold
                    if drift:
                        drift_found = True
                    drift_report[column] = {
                        'ks_statistic': ks_statistic,
                        'p_value': p_value,
                        'drift_found': drift
                    }
            drift_report['overall_drift_found'] = drift_found
            return drift_report
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def validate_data(self) -> DataValidationArtifact:
        try:
            # Load the schema
            schema = self.schema
            logging.info(f"Schema loaded: {schema}")

            # Load the train and test data
            train_data = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Validate the schema
            self.validate_schema(train_data, schema)
            self.validate_schema(test_data, schema)

            # Check for data drift
            drift_report = self.check_data_drift(train_data, test_data, schema)

            # Save the drift report
            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)
            drift_report_file_path = os.path.join(self.data_validation_config.drift_report_dir, "drift_report.yaml")
            save_yaml_file(drift_report_file_path, drift_report)
            
            data_validation_artifact = DataValidationArtifact(
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomerChurnException(e, sys) from e