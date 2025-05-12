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