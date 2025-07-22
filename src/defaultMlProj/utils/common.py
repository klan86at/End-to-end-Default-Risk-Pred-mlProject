import os
from box.exceptions import BoxValueError
from box import ConfigBox
import yaml
from defaultMlProj import logger
from pathlib import Path
import json
import joblib
from ensure import ensure_annotations
from typing import Any, Union, Dict, List

@ensure_annotations
def read_yaml(path_to_yaml: Union[str, Path]) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (Union[str, Path]): Path to the YAML file.
        
    Raises:
        ValueError: if yaml file is empty or not found.
        e: empty file
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: List, verbose= True):
    """
    Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of paths to the directories to be created.
        ignore_log(bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")

@ensure_annotations
def save_json(path: Union[str, Path], data: Dict):
    """
    Saves a dictionary to a JSON file.
    
    Args:
        path (Union[str, Path]): Path to the JSON file.
        data (Dict): Data to be saved in json file.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at {path}")


@ensure_annotations
def load_json(path: Union[str, Path]) -> ConfigBox:
    """
    Loads json file data.
    
    Args:
        path (Union[str, Path]): Path to the JSON file.
        
    Returns:
        ConfigBox: Data as class attributes instead of dict.
    """
    with open(path) as f:
        content = json.load(f)
        logger.info(f"JSON file loaded successfully from {path}")
        return ConfigBox(content)
    

@ensure_annotations
def save_bin(data: Any, path: Union[str, Path]):
    """
    Saves data to a binary file using joblib.
    
    Args:
        data (Any): Data to be saved as binary.
        path (Union[str, Path]): Path to the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Data saved as binary file at: {path}")


@ensure_annotations
def load_bin(path: Union[str, Path]) -> Any:
    """
    Loads data from a binary file using joblib.
    
    Args:
        path (Union[str, Path]): Path to the binary file.
        
    Returns:
        Any: Data loaded from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Data loaded from binary file at: {path}")
    return data


@ensure_annotations
def get_size(path: Union[str, Path]) -> str:
    """
    Gets the size of a file in KB.
    
    Args:
        path (Union[str, Path]): Path to the file.
        
    Returns:
        str: Size of the file in KB.
    """
    Size_in_KB = round(os.path.getsize(path) / 1024)
    return f"~ {Size_in_KB} KB"