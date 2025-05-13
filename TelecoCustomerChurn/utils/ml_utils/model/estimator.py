from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.constants.training_pipeline import MODEL_TRAINING_MODEL_DIR, MODEL_TRAINING_MODEL_FILE_NAME
import os
import sys

class TelecoCustomerChurnModel:
    def __init__(self,preprocessed_object, model):
        try:
            self.preprocessed_object = preprocessed_object
            self.model = model
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def predict(self, data):
        try:
            data = self.preprocessed_object.transform(data)
            predictions = self.model.predict(data)
            return predictions
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
