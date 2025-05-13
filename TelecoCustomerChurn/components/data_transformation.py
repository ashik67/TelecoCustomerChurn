from TelecoCustomerChurn.entity.config_entity import DataTransformationConfig
from TelecoCustomerChurn.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.utils.main_utils import save_numpy_array_data,save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys
import os
import pandas as pd
import numpy as np

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            logging.info("Data Transformation component initialized.")
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process.")
            # Load the validated train and test data
            logging.info(f"Loading validated train data from: {self.data_validation_artifact.valid_train_file_path}")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            logging.info(f"Loading validated test data from: {self.data_validation_artifact.valid_test_file_path}")
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)
            logging.info(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")
            # Check if the target column is present in both train and test data
            if self.data_transformation_config.target_column_name not in train_df.columns or self.data_transformation_config.target_column_name not in test_df.columns:
                logging.error(f"Target column '{self.data_transformation_config.target_column_name}' is missing in train or test data.")
                raise ValueError(f"Target column '{self.data_transformation_config.target_column_name}' is missing in train or test data.")
            # Separate features and target variable
            logging.info(f"Separating features and target column: {self.data_transformation_config.target_column_name}")
            X_train = train_df.drop(columns=[self.data_transformation_config.target_column_name])
            y_train = train_df[self.data_transformation_config.target_column_name]  
            X_test = test_df.drop(columns=[self.data_transformation_config.target_column_name])
            y_test = test_df[self.data_transformation_config.target_column_name]
            # Check for missing values in the target variable
            if y_train.isnull().any() or y_test.isnull().any():
                logging.error("Missing values found in the target variable.")
                raise ValueError("Missing values found in the target variable.")
            # Define categorical and numerical columns
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {numerical_cols}")
            # Remove the target column from numerical columns
            if self.data_transformation_config.target_column_name in numerical_cols:
                numerical_cols.remove(self.data_transformation_config.target_column_name)
            # Check if categorical and numerical columns are present
            if not categorical_cols and not numerical_cols:
                logging.error("No categorical or numerical columns found in the dataset.")
                raise ValueError("No categorical or numerical columns found in the dataset.")
            # Define the preprocessing steps for numerical and categorical columns
            logging.info(f"Numeric imputer params: {self.data_transformation_config.numeric_imputer_params}")
            logging.info(f"Categorical imputer params: {self.data_transformation_config.categorical_imputer_params}")
            numerical_transformer = Pipeline(steps=[
                ('imputer', KNNImputer(**self.data_transformation_config.numeric_imputer_params)),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(**self.data_transformation_config.categorical_imputer_params)),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            # Combine the preprocessing steps using ColumnTransformer
            logging.info("Creating ColumnTransformer for preprocessing.")
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )   
            # Ensure output directories exist before saving files
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_target_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_target_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessed_object_file_path), exist_ok=True)
            # Fit and transform the training data
            logging.info("Fitting and transforming training data.")
            X_train_transformed = preprocessor.fit_transform(X_train)
            # Transform the test data
            logging.info("Transforming test data.")
            X_test_transformed = preprocessor.transform(X_test)
            # Ensure dense arrays for saving
            if hasattr(X_train_transformed, 'toarray'):
                X_train_transformed = X_train_transformed.toarray()
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            # Encode the target variable ('Churn') to numeric: 0 for 'No', 1 for 'Yes'
            logging.info("Encoding target variable to numeric: 0 for 'No', 1 for 'Yes'.")
            y_train_encoded = y_train.map({'No': 0, 'Yes': 1}).astype(np.int8)
            y_test_encoded = y_test.map({'No': 0, 'Yes': 1}).astype(np.int8)
            # Save the encoded target arrays as numpy arrays
            logging.info(f"Saving encoded train target to: {self.data_transformation_config.transformed_train_target_file_path}")
            save_numpy_array_data(self.data_transformation_config.transformed_train_target_file_path, y_train_encoded.values)
            logging.info(f"Saving encoded test target to: {self.data_transformation_config.transformed_test_target_file_path}")
            save_numpy_array_data(self.data_transformation_config.transformed_test_target_file_path, y_test_encoded.values)
            # Save the transformed data as numpy arrays
            logging.info(f"Saving transformed train data to: {self.data_transformation_config.transformed_train_file_path}")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, X_train_transformed)
            logging.info(f"Saving transformed test data to: {self.data_transformation_config.transformed_test_file_path}")
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, X_test_transformed)
            # Save the preprocessor object 
            logging.info(f"Saving preprocessor object to: {self.data_transformation_config.preprocessed_object_file_path}")
            save_object(self.data_transformation_config.preprocessed_object_file_path, preprocessor)
            # Create and return the DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_target_file_path=self.data_transformation_config.transformed_train_target_file_path,
                transformed_test_target_file_path=self.data_transformation_config.transformed_test_target_file_path,
                preprocessed_object_file_path=self.data_transformation_config.preprocessed_object_file_path
            )
            logging.info("Data transformation process completed successfully.")
            return data_transformation_artifact
        except Exception as e:
            logging.error(f"Exception during data transformation: {e}")
            raise CustomerChurnException(e, sys) from e
