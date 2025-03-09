# Import standard libraries for testing and mocking
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from unittest.mock import patch, MagicMock

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

# Suppress logging output during tests to keep the console clean
logging.getLogger().setLevel(logging.CRITICAL)

# Sample dataset for testing, created once and reused across tests
# This mimics a credit scoring dataset with numerical, categorical, and datetime features
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

# Test functions with detailed documentation

def test_preprocess_data():
    """Test the preprocess_data function to ensure it processes data correctly."""
    # Make a copy of the sample data to avoid modifying the original
    processed_data = preprocess_data(data.copy())
    # Check if preprocessing succeeded (not None)
    assert processed_data is not None, "Preprocessing returned None"
    # Verify no missing values remain after handling
    assert not processed_data.isnull().any().any(), "Processed data contains missing values"
    # Confirm unnecessary columns are dropped
    assert 'TransactionId' not in processed_data.columns, "TransactionId was not dropped"
    # Check if datetime features are extracted
    assert 'TransactionHour' in processed_data.columns, "TransactionHour feature not created"
    # Ensure categorical encoding worked (e.g., 'CurrencyCode' becomes dummy variables)
    assert 'CurrencyCode_EUR' in processed_data.columns, "Categorical encoding failed for CurrencyCode"
    print("test_preprocess_data: PASSED")

def test_split_the_data():
    """Test the split_the_data function to verify correct data splitting."""
    # Preprocess the data first
    processed_data = preprocess_data(data.copy())
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_the_data(processed_data, target_column='Label', test_size=0.2)
    # Ensure all returned objects are not None
    assert X_train is not None, "X_train is None"
    assert X_test is not None, "X_test is None"
    assert y_train is not None, "y_train is None"
    assert y_test is not None, "y_test is None"
    # Verify the split proportions (80% train, 20% test)
    assert len(X_train) == int(0.8 * len(processed_data)), "Training set size incorrect"
    assert len(X_test) == int(0.2 * len(processed_data)), "Test set size incorrect"
    print("test_split_the_data: PASSED")

def test_define_models():
    """Test the define_models function to ensure models are correctly defined."""
    # Get the models dictionary
    models = define_models()
    # Check if expected models are present
    assert 'LogisticRegression' in models, "LogisticRegression not in models"
    assert 'RandomForest' in models, "RandomForest not in models"
    # Verify model types
    assert isinstance(models['LogisticRegression'], Pipeline), "LogisticRegression is not a Pipeline"
    assert isinstance(models['RandomForest'], RandomForestClassifier), "RandomForest is not a RandomForestClassifier"
    print("test_define_models: PASSED")

def test_define_hyperparameter_grids():
    """Test the define_hyperparameter_grids function for correct parameter grids."""
    # Get the parameter grids
    param_grids = define_hyperparameter_grids()
    # Ensure grids exist for both models
    assert 'LogisticRegression' in param_grids, "LogisticRegression not in param_grids"
    assert 'RandomForest' in param_grids, "RandomForest not in param_grids"
    # Check key hyperparameters are included
    assert 'logistic__C' in param_grids['LogisticRegression'], "C parameter missing for LogisticRegression"
    assert 'n_estimators' in param_grids['RandomForest'], "n_estimators missing for RandomForest"
    print("test_define_hyperparameter_grids: PASSED")

@patch('sklearn.model_selection.GridSearchCV')
def test_perform_grid_search(mock_grid_search):
    """Test the perform_grid_search function with mocked GridSearchCV."""
    # Mock GridSearchCV to avoid actual computation
    mock_instance = MagicMock()
    mock_instance.best_estimator_ = LogisticRegression()
    mock_grid_search.return_value = mock_instance
    # Prepare data and inputs
    models = define_models()
    param_grids = define_hyperparameter_grids()
    processed_data = preprocess_data(data.copy())
    X_train, _, y_train, _ = split_the_data(processed_data)
    # Run grid search
    best_models = perform_grid_search(models, param_grids, X_train, y_train)
    # Verify results
    assert best_models is not None, "perform_grid_search returned None"
    assert 'LogisticRegression' in best_models, "LogisticRegression missing from best_models"
    assert 'RandomForest' in best_models, "RandomForest missing from best_models"
    print("test_perform_grid_search: PASSED")

def test_evaluate_best_models():
    """Test the evaluate_best_models function with fitted models."""
    # Preprocess and split data
    processed_data = preprocess_data(data.copy())
    X_train, X_test, y_train, y_test = split_the_data(processed_data)
    # Create and fit dummy models
    best_models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(solver='liblinear', random_state=42))
        ]).fit(X_train, y_train),
        'RandomForest': RandomForestClassifier(random_state=42).fit(X_train, y_train)
    }
    # Evaluate models
    results = evaluate_best_models(best_models, X_test, y_test)
    # Check results
    assert results is not None, "evaluate_best_models returned None"
    assert 'LogisticRegression' in results, "LogisticRegression missing from results"
    assert 'RandomForest' in results, "RandomForest missing from results"
    assert 'Accuracy' in results['LogisticRegression'], "Accuracy missing for LogisticRegression"
    print("test_evaluate_best_models: PASSED")

@patch('joblib.dump')
@patch('os.makedirs')
def test_save_best_models(mock_makedirs, mock_joblib_dump):
    """Test the save_best_models function with mocked file operations."""
    # Create dummy models
    best_models = {
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }
    # Run save function
    save_best_models(best_models, save_dir='test_models')
    # Verify directory creation and file saving
    mock_makedirs.assert_called_once_with('test_models')
    assert mock_joblib_dump.call_count == 2, "joblib.dump not called twice"
    print("test_save_best_models: PASSED")

# Custom test runner to execute all tests and report results
def run_tests():
    """Run all test functions and report results."""
    tests = [
        test_preprocess_data,
        test_split_the_data,
        test_define_models,
        test_define_hyperparameter_grids,
        test_perform_grid_search,
        test_evaluate_best_models,
        test_save_best_models
    ]
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"{test.__name__}: FAILED - {str(e)}")
        except Exception as e:
            print(f"{test.__name__}: ERROR - {str(e)}")
    
    print(f"\nRan {total} tests: {passed} passed, {total - passed} failed")

if __name__ == '__main__':
    run_tests()