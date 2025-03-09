# Import standard libraries for testing and mocking
import sys
import os
import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch, MagicMock
import logging

# Dynamically add the 'scripts' folder to sys.path to resolve import issues
# This ensures the Modelling script in the scripts folder is accessible from the tests folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

# Import functions from the Modelling script (assumed to be named 'Modelling.py' in the scripts folder)
from Modelling import (
    preprocess_data,
    split_the_data,
    define_models,
    define_hyperparameter_grids,
    perform_grid_search,
    evaluate_best_models,
    save_best_models
)

# Fixture for sample dataset, reusable across tests
@pytest.fixture
def sample_data():
    """Create a sample dataset mimicking credit scoring data."""
    data = pd.DataFrame({
        'TransactionId': [1, 2, 3, 4],
        'BatchId': [101, 102, 103, 104],
        'AccountId': [1001, 1002, 1003, 1004],
        'SubscriptionId': [2001, 2002, 2003, 2004],
        'CustomerId': [3001, 3002, 3003, 3004],
        'CurrencyCode': ['USD', 'EUR', 'USD', 'EUR'],
        'CountryCode': ['US', 'FR', 'US', 'FR'],
        'ProductId': [1, 2, 1, 2],
        'ChannelId': [1, 2, 1, 2],
        'TransactionStartTime': ['2023-01-01 12:00:00', '2023-01-02 13:00:00',
                                 '2023-01-03 14:00:00', '2023-01-04 15:00:00'],
        'Amount': [100.0, 200.0, np.nan, 400.0],
        'Label': [0, 1, 0, 1]
    })
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    return data

# Fixture for logging setup
@pytest.fixture
def setup_logging():
    """Set up a logger for tests to capture output without cluttering console."""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)  # Suppress logs during tests
    return logger

def test_preprocess_data(sample_data, setup_logging):
    """Test preprocess_data for correct handling of duplicates, missing values, and encoding."""
    processed_data = preprocess_data(sample_data.copy())
    assert processed_data is not None, "Preprocessing returned None"
    assert not processed_data.isnull().any().any(), "Processed data contains missing values"
    assert 'TransactionId' not in processed_data.columns, "TransactionId was not dropped"
    assert 'TransactionHour' in processed_data.columns, "TransactionHour feature not created"
    # Check if categorical encoding worked for CurrencyCode
    assert 'CurrencyCode_EUR' in processed_data.columns, "Categorical encoding failed for CurrencyCode"

def test_split_the_data(sample_data, setup_logging):
    """Test split_the_data for correct splitting into train and test sets."""
    processed_data = preprocess_data(sample_data.copy())
    X_train, X_test, y_train, y_test = split_the_data(processed_data, target_column='Label', test_size=0.2)
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    assert len(X_train) == int(0.8 * len(processed_data)), "Training set size incorrect"
    assert len(X_test) == int(0.2 * len(processed_data)), "Test set size incorrect"

def test_define_models(setup_logging):
    """Test define_models for correct model instantiation."""
    models = define_models()
    assert 'LogisticRegression' in models, "LogisticRegression not in models"
    assert 'RandomForest' in models, "RandomForest not in models"
    assert isinstance(models['LogisticRegression'], Pipeline), "LogisticRegression is not a Pipeline"
    assert isinstance(models['RandomForest'], RandomForestClassifier), "RandomForest is not a RandomForestClassifier"

def test_define_hyperparameter_grids(setup_logging):
    """Test define_hyperparameter_grids for correct parameter grid setup."""
    param_grids = define_hyperparameter_grids()
    assert 'LogisticRegression' in param_grids, "LogisticRegression not in param_grids"
    assert 'RandomForest' in param_grids, "RandomForest not in param_grids"
    assert 'logistic__C' in param_grids['LogisticRegression'], "C parameter missing for LogisticRegression"
    assert 'n_estimators' in param_grids['RandomForest'], "n_estimators missing for RandomForest"

@patch('sklearn.model_selection.GridSearchCV')
def test_perform_grid_search(sample_data, mock_grid_search, setup_logging):
    """Test perform_grid_search with mocked GridSearchCV to avoid actual fitting."""
    mock_instance = MagicMock()
    mock_instance.best_estimator_ = LogisticRegression()
    mock_grid_search.return_value = mock_instance
    models = define_models()
    param_grids = define_hyperparameter_grids()
    processed_data = preprocess_data(sample_data.copy())
    X_train, _, y_train, _ = split_the_data(processed_data)
    best_models = perform_grid_search(models, param_grids, X_train, y_train)
    assert best_models is not None, "perform_grid_search returned None"
    assert 'LogisticRegression' in best_models, "LogisticRegression missing from best_models"
    assert 'RandomForest' in best_models, "RandomForest missing from best_models"

def test_evaluate_best_models(sample_data, setup_logging):
    """Test evaluate_best_models with fitted models on test data."""
    processed_data = preprocess_data(sample_data.copy())
    X_train, X_test, y_train, y_test = split_the_data(processed_data)
    best_models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(solver='liblinear', random_state=42))
        ]).fit(X_train, y_train),
        'RandomForest': RandomForestClassifier(random_state=42).fit(X_train, y_train)
    }
    results = evaluate_best_models(best_models, X_test, y_test)
    assert results is not None, "evaluate_best_models returned None"
    assert 'LogisticRegression' in results, "LogisticRegression missing from results"
    assert 'RandomForest' in results, "RandomForest missing from results"
    assert 'Accuracy' in results['LogisticRegression'], "Accuracy missing for LogisticRegression"

@patch('joblib.dump')
@patch('os.makedirs')
def test_save_best_models(mock_makedirs, mock_joblib_dump, setup_logging):
    """Test save_best_models with mocked file operations."""
    best_models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }
    save_best_models(best_models, save_dir='test_models')
    mock_makedirs.assert_called_once_with('test_models')
    assert mock_joblib_dump.call_count == 2, "joblib.dump not called twice"