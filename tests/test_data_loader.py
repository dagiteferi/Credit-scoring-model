import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
import logging
from src.data_loader import load_data, create_directory_if_not_exists

# Sample data for mocking file reads
SAMPLE_CSV = "customer_id,income,credit_score\n1,50000,700\n2,60000,650\n"

@pytest.fixture
def setup_logging():
    """Fixture to set up logging for tests."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def test_create_directory_if_not_exists_new_dir(tmp_path, setup_logging):
    """Test creating a new directory."""
    dir_path = tmp_path / "new_dir"
    create_directory_if_not_exists(str(dir_path))
    assert os.path.exists(dir_path)
    assert setup_logging.handlers  # Ensure logging is active

def test_create_directory_if_not_exists_existing_dir(tmp_path, setup_logging):
    """Test handling an existing directory."""
    dir_path = tmp_path / "existing_dir"
    os.makedirs(dir_path)
    create_directory_if_not_exists(str(dir_path))
    assert os.path.exists(dir_path)

def test_load_data_success(tmp_path):
    """Test loading data from an existing CSV file."""
    file_path = tmp_path / "data.csv"
    file_path.write_text(SAMPLE_CSV)
    data = load_data(file_path.name, dataset_dir=str(tmp_path))
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2, 3)
    assert list(data.columns) == ["customer_id", "income", "credit_score"]

def test_load_data_file_not_found(tmp_path):
    """Test loading data when the file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv", dataset_dir=str(tmp_path))

def test_load_data_invalid_csv(tmp_path):
    """Test loading data from an invalid CSV."""
    file_path = tmp_path / "invalid.csv"
    file_path.write_text("invalid,data\nno,structure")
    with pytest.raises(Exception):
        load_data(file_path.name, dataset_dir=str(tmp_path))