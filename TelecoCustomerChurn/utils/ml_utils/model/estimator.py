from TelecoCustomerChurn.logging.logger import logging
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.constants.training_pipeline import MODEL_TRAINING_MODEL_DIR, MODEL_TRAINING_MODEL_FILE_NAME
import os
import sys
import pandas as pd
import numpy as np

class TelecoCustomerChurnModel:
    def __init__(self,preprocessed_object, model):
        try:
            self.preprocessed_object = preprocessed_object
            self.model = model
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
        
    def predict(self, data):
        try:
            # Only enforce column names if input is a DataFrame
            if hasattr(self.preprocessed_object, 'named_steps') and 'preprocessor' in self.preprocessed_object.named_steps:
                preprocessor = self.preprocessed_object.named_steps['preprocessor']
                if hasattr(preprocessor, 'feature_names_in_'):
                    expected_cols = list(preprocessor.feature_names_in_)
                    if isinstance(data, pd.DataFrame):
                        missing = [col for col in expected_cols if col not in data.columns]
                        extra = [col for col in data.columns if col not in expected_cols]
                        if missing:
                            raise CustomerChurnException(
                                f"Input DataFrame is missing columns: {missing}", sys
                            )
                        if extra:
                            # Optionally, drop extra columns
                            data = data[expected_cols]
                        else:
                            data = data[expected_cols]
                    elif not isinstance(data, np.ndarray):
                        raise CustomerChurnException(
                            f"Input data must be a pandas DataFrame or numpy array.", sys
                        )
            # Let the pipeline handle all further transformation/feature selection
            data_transformed = self.preprocessed_object.transform(data)
            predictions = self.model.predict(data_transformed)
            return predictions
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
