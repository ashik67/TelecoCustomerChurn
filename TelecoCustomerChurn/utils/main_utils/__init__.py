import yaml
from TelecoCustomerChurn.exception.exception import CustomerChurnException
from TelecoCustomerChurn.logging.logger import logging
import os
import sys
import numpy as np
import pandas as pd
import dill
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