import pandas as pd
import os
import logging
import sys

# Add the parent directory to the system path to allow module imports
sys.path.append(os.path.join(os.path.abspath('../')))

# Set up logging to track events and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs")), 'data_loading.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Define the default dataset directory
_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it does not already exist.

    Parameters:
    -----------
    directory_path (str): The path of the directory to create.
    """
    # Check if the directory exists; create it if it doesn't
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

def load_data(file_path, dataset_dir=_DATASET_DIR):
    """
    Load a dataset from a CSV file and return it as a pandas DataFrame.

    This function constructs the full file path, checks for the file's existence,
    and loads the data while logging key events. It’s designed for robustness and
    ease of debugging.

    Parameters:
    -----------
    file_path (str): The relative path to the dataset file (e.g., 'credit_data.csv').
    dataset_dir (str): The root directory where the dataset is stored. Defaults to _DATASET_DIR.

    Returns:
    --------
    pd.DataFrame: The loaded dataset.

    Raises:
    -------
    FileNotFoundError: If the specified file is not found in the dataset directory.
    Exception: If an error occurs while loading the file (e.g., corrupted file).
    """
    # Construct the full file path from the dataset directory and file name
    full_file_path = os.path.join(dataset_dir, file_path)

    # Verify that the file exists before attempting to load it
    if not os.path.exists(full_file_path):
        error_msg = f"File not found: {full_file_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Log the attempt to load the data and perform the loading
        logging.info(f"Attempting to load data from: {full_file_path}")
        data = pd.read_csv(full_file_path)
        logging.info(f"Successfully loaded data from: {full_file_path}")
        return data
    except Exception as e:
        # Log any unexpected errors with details for debugging
        error_msg = f"Error loading data from {full_file_path}: {e}"
        logging.error(error_msg)
        raise e

# Ensure required directories exist before proceeding
create_directory_if_not_exists(_DATASET_DIR)
create_directory_if_not_exists(os.path.join(os.path.dirname(__file__), "../logs"))