import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

import certifi
ca=certifi.where()

import pandas as pd
from pymongo import MongoClient 
import numpy as np
from TelecoCustomerChurn.exception.exception import CustomerChurnException as TelecoCustomerChurnException
from TelecoCustomerChurn.logging.logger import logging

class DataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise TelecoCustomerChurnException(e, sys)
        
    def csv_to_json(self, file_path):
        try:
            df = pd.read_csv(file_path)
            json_data = df.to_json(orient='records')
            return json_data
        except Exception as e:
            raise TelecoCustomerChurnException(e, sys)
    
    def push_data_to_mongo(self, json_data, database_name, collection_name):
        try:
            client = MongoClient(MONGO_URI, tlsCAFile=ca)
            db = client[database_name]
            collection = db[collection_name]

            data = json.loads(json_data)

            if not data:
                logging.warning("No data to insert into MongoDB.")
                return

            if isinstance(data, list):
                result = collection.insert_many(data)
                logging.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}.")
            else:
                result = collection.insert_one(data)
                logging.info(f"Inserted 1 document into {collection_name}.")

        except Exception as e:
            raise TelecoCustomerChurnException(e, sys)
        finally:
            client.close()



if __name__ == "__main__":
    try:
        data_extractor = DataExtract()
        file_path = "Data/TelecoCustomerChurn.csv"
        json_data = data_extractor.csv_to_json(file_path)
        data_extractor.push_data_to_mongo(json_data, "TelecomCustomerChurn", "TelecomCustomerChurn")
        logging.info("Data pushed to MongoDB successfully.")
    except Exception as e:
        logging.error(f"Error: {e}")