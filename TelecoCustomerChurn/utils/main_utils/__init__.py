import yaml
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
import os
import sys
import numpy as np
import pandas as pd
import pickle

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Content of the YAML file.
    """
    try:
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
        return content
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    

    
def save_yaml_file(file_path: str, content: dict) -> None:
    """
    Saves a dictionary to a YAML file.

    Args:
        file_path (str): Path to the YAML file.
        content (dict): Content to be saved.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    


def convert_numpy_types(obj):
    import numpy as np
    import math
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.generic, np.bool_)):
        v = obj.item()
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, str) and obj.strip().lower() in {'.nan', 'nan'}:
        return None
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
    

    
def save_object(file_path: str, obj) -> None:
    """
    Saves an object to a file using pickle.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj: The object to be saved.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    

    
def load_object(file_path: str):
    """
    Loads an object from a file using pickle.

    Args:
        file_path (str): Path to the file from which the object will be loaded.

    Returns:
        The loaded object.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    

    
def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a file.

    Args:
        file_path (str): Path to the file where the array will be saved.
        array (np.ndarray): The NumPy array to be saved.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    

    
def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.

    Args:
        file_path (str): Path to the file from which the array will be loaded.

    Returns:
        np.ndarray: The loaded NumPy array.
    """
    try:
        # Allow loading of object arrays (e.g., if categorical or mixed types slipped through)
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param_grids):
    """
    Evaluates multiple models with their parameter grids and returns a report containing all metrics and best params for each model.

    Args:
        X_train (numpy.ndarray): Feature data for training.
        y_train (numpy.ndarray): Target data for training.
        X_test (numpy.ndarray): Feature data for testing.
        y_test (numpy.ndarray): Target data for testing.
        models (dict): Dictionary of model names and their corresponding model objects.
        param_grids (dict): Dictionary of model names and their corresponding param grids.

    Returns:
        dict: A report containing model names, their metrics, best params, and the best model overall.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    try:
        report = {}
        best_model_name = None
        best_score = -float('inf')
        best_model_metrics = None
        best_model_params = None
        for model_name, model in models.items():
            param_grid = param_grids.get(model_name, {})
            # Use F1 score for model selection and hyperparameter tuning
            grid = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            y_pred = best_estimator.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred),
                'best_params': grid.best_params_
            }
            report[model_name] = metrics
            # Select best model by F1 score
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model_name = model_name
                best_model_metrics = metrics
                best_model_params = grid.best_params_
        report['best_model'] = best_model_name
        report['best_model_metrics'] = best_model_metrics
        report['best_model_params'] = best_model_params
        return report
    except Exception as e:
        raise CustomerChurnException(e, sys) from e