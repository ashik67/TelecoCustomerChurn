from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.entity.config_entity import DataValidationConfig
from TelecoCustomerChurn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from TelecoCustomerChurn.constants.training_pipeline import SCHEMA_FILE_PATH
from TelecoCustomerChurn.utils.main_utils import read_yaml_file, save_yaml_file, convert_numpy_types
import os
import sys
import pandas as pd
from typing import List
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            logging.info("Initializing Data Validation component.")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def validate_schema(self, data: pd.DataFrame, schema: dict) -> bool:
        """
        Validate the schema of the DataFrame against the provided schema.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            schema (dict): The schema to validate against.

        Returns:
            bool: True if schema is valid, False otherwise.

        Raises:
            ValueError: If the DataFrame does not match the schema.
        """
        try:
            # Mapping from schema type to acceptable pandas dtypes
            type_mapping = {
                'string': ['object', 'string'],
                'int': ['int64', 'int32', 'int16', 'int8'],
                'float': ['float64', 'float32', 'float16']
            }
            for column, properties in schema["columns"].items():
                if column not in data.columns:
                    logging.error(f"Column '{column}' is missing from the DataFrame.")
                    return False
                expected_type = properties['type']
                pandas_dtype = str(data[column].dtype)
                if expected_type in type_mapping:
                    if pandas_dtype not in type_mapping[expected_type]:
                        logging.error(f"Column '{column}' has incorrect type. Expected {expected_type}, got {pandas_dtype}.")
                        return False
                else:
                    if pandas_dtype != expected_type:
                        logging.error(f"Column '{column}' has incorrect type. Expected {expected_type}, got {pandas_dtype}.")
                        return False
                if 'min' in properties and data[column].min() < properties['min']:
                    logging.error(f"Column '{column}' has values less than minimum allowed value {properties['min']}.")
                    return False
                if 'max' in properties and data[column].max() > properties['max']:
                    logging.error(f"Column '{column}' has values greater than maximum allowed value {properties['max']}.")
                    return False
            logging.info("Schema validation passed.")
            return True
        except Exception as e:
            logging.error(f"Exception during schema validation: {e}")
            raise CustomerChurnException(e, sys) from e
    
    def check_data_drift(self, train_data: pd.DataFrame, test_data: pd.DataFrame, schema: dict) -> dict:
        """
        Check for data drift between the train and test datasets.
        Impute missing values before running the drift check.
        """
        try:
            drift_report = {}
            drift_found = False
            threshold = 0.05  # p-value threshold for drift
            for column, properties in schema["columns"].items():
                if column in train_data.columns and column in test_data.columns:
                    # Impute missing values
                    if properties['type'] in ['float', 'int']:
                        train_col = train_data[column].fillna(train_data[column].median())
                        test_col = test_data[column].fillna(test_data[column].median())
                    else:
                        train_col = train_data[column].fillna(train_data[column].mode()[0] if not train_data[column].mode().empty else "Unknown")
                        test_col = test_data[column].fillna(test_data[column].mode()[0] if not test_data[column].mode().empty else "Unknown")
                    train_missing = train_data[column].isnull().sum()
                    test_missing = test_data[column].isnull().sum()
                    total_train = len(train_data[column])
                    total_test = len(test_data[column])
                    logging.info(f"Column '{column}': missing in train={train_missing}/{total_train}, missing in test={test_missing}/{total_test}")
                    ks_statistic, p_value = ks_2samp(train_col, test_col)
                    drift = p_value < threshold
                    if drift:
                        drift_found = True
                        logging.warning(f"Data drift detected in column '{column}': KS={ks_statistic}, p-value={p_value}")
                    else:
                        logging.info(f"No significant drift in column '{column}': KS={ks_statistic}, p-value={p_value}")
                    drift_report[column] = {
                        'ks_statistic': ks_statistic,
                        'p_value': p_value,
                        'drift_found': drift
                    }
            drift_report['overall_drift_found'] = drift_found
            if drift_found:
                logging.warning("Overall data drift detected in one or more columns.")
            else:
                logging.info("No overall data drift detected.")
            return drift_report
        except Exception as e:
            logging.error(f"Exception during data drift check: {e}")
            raise CustomerChurnException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process.")
            # Load the schema
            schema = self.schema
            logging.info(f"Schema loaded: {schema}")

            # Load the train and test data
            train_data = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_data = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Loaded train data from {self.data_ingestion_artifact.train_file_path} with shape {train_data.shape}")
            logging.info(f"Loaded test data from {self.data_ingestion_artifact.test_file_path} with shape {test_data.shape}")

            # Ensure TotalCharges is float for validation
            if 'TotalCharges' in train_data.columns:
                train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'], errors='coerce')
            if 'TotalCharges' in test_data.columns:
                test_data['TotalCharges'] = pd.to_numeric(test_data['TotalCharges'], errors='coerce')

            # Validate the schema
            train_schema_valid = self.validate_schema(train_data, schema)
            test_schema_valid = self.validate_schema(test_data, schema)
            schema_status = train_schema_valid and test_schema_valid
            if schema_status:
                logging.info("Both train and test data passed schema validation.")
            else:
                logging.error("Schema validation failed for train or test data.")

            # Check for data drift
            drift_report = self.check_data_drift(train_data, test_data, schema)

            # Save the drift report
            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)
            drift_report_file_path = os.path.join(self.data_validation_config.drift_report_dir, "drift_report.yaml")
            drift_report = convert_numpy_types(drift_report)
            save_yaml_file(drift_report_file_path, drift_report)
            logging.info(f"Drift report saved at {drift_report_file_path}")

            validation_status = schema_status and not drift_report['overall_drift_found']

            # Save valid or invalid data based on validation_status
            if validation_status:
                os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.data_validation_config.valid_test_file_path), exist_ok=True)
                train_data.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_data.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                logging.info(f"Valid train data saved at {self.data_validation_config.valid_train_file_path}")
                logging.info(f"Valid test data saved at {self.data_validation_config.valid_test_file_path}")
            else:
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path), exist_ok=True)
                train_data.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_data.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                logging.warning(f"Invalid train data saved at {self.data_validation_config.invalid_train_file_path}")
                logging.warning(f"Invalid test data saved at {self.data_validation_config.invalid_test_file_path}")

            data_validation_artifact = DataValidationArtifact(
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=drift_report_file_path,
                validation_status=validation_status
            )

            logging.info(f"Data validation artifact created: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            logging.error(f"Exception during data validation: {e}")
            raise CustomerChurnException(e, sys) from e