from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging

from TelecoCustomerChurn.entity.config_entity import DataIngestionConfig
from TelecoCustomerChurn.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

MONGO_CLIENT = pymongo.MongoClient(os.getenv("MONGO_URI"))

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Exporting data from MongoDB to DataFrame")
            # Export data from MongoDB to DataFrame
            df = pd.DataFrame(list(MONGO_CLIENT[self.data_ingestion_config.database_name][self.data_ingestion_config.collection_name].find()))
            logging.info(f"DataFrame shape: {df.shape}")

            # Drop the '_id' column if it exists
            if '_id' in df.columns:
                df.drop(columns=['_id'], inplace=True)
            logging.info(f"DataFrame shape after dropping '_id': {df.shape}")
            
            if df.empty:
                raise ValueError("The DataFrame is empty. Please check the MongoDB collection.")
            # Check for duplicate rows
            if df.duplicated().any():
                df.drop_duplicates(inplace=True)
                logging.info("Duplicate rows removed from the DataFrame")
            df.replace({'na':np.nan}, inplace=True)
            if df.isnull().values.any():
                df.dropna(inplace=True)

            # Save the cleaned DataFrame to feature_store before splitting
            feature_store_dir = os.path.join(self.data_ingestion_config.data_ingestion_dir, "feature_store")
            os.makedirs(feature_store_dir, exist_ok=True)
            feature_store_file = os.path.join(feature_store_dir, "cleaned_data.csv")
            df.to_csv(feature_store_file, index=False)
            logging.info(f"Feature store file saved at: {feature_store_file}")

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)
            logging.info(f"Train set shape: {train_set.shape}, Test set shape: {test_set.shape}")

            # Save the train and test sets to CSV files
            os.makedirs(self.data_ingestion_config.ingested_dir, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False)
            logging.info(f"Train file saved at: {self.data_ingestion_config.train_file_path}")
            logging.info(f"Test file saved at: {self.data_ingestion_config.test_file_path}")
            logging.info("Data ingestion completed successfully")
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
