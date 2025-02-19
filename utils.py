# clinical_trial_api/app/utils.py

__author__ = "vishal"

"""
Utility functions for the Clinical Trial Retrieval API.
"""

import json
import logging
from typing import Dict, Any, List
import os


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.
    
    :param file_path: Path to the JSON file.
    :return: Dictionary containing the JSON data.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        logging.info(f"Loaded JSON file from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file {file_path}.")
        return {}
    except Exception as e:
        logging.exception(f"Error loading JSON file {file_path}: {e}")
        return {}


def load_trial_info(nctids: List[str], file_path: str = "dataset/corpus.jsonl") -> Dict[str, Any]:
    """
    Load trial data from a JSONL file into a dictionary for specific NCTIDs.
    
    :param nctids: List of NCTIDs to retrieve trial details for.
    :param file_path: Path to the JSONL file (default: "dataset/corpus.jsonl").
    :return: Dictionary where keys are NCTIDs and values are trial details.
    """
    trial_info = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                trial = json.loads(line)
                trial_id = trial.get("_id")  # Assuming each line has an "_id" field for trial ID
                if trial_id in nctids:
                    trial_info[trial_id] = trial
        logging.info(f"Loaded trial data for {len(trial_info)} out of {len(nctids)} NCTIDs from {file_path}.")
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON format in file {file_path}.")
    except Exception as e:
        logging.exception(f"Error loading trial data from {file_path}: {e}")
    return trial_info

