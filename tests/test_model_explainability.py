import pytest
import pandas as pd
import os
import logging
import joblib
import numpy as np
from unittest.mock import Mock, patch
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Use relative import from the week-6 subdirectory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'week-6')))
from scripts.model_explainability import (
    load_models,
    load_or_preprocess_data,
    explain_logistic_regression,
    explain_random_forest,
    run_explainability_pipeline
)

# Updated sample data with more rows to support stratification
SAMPLE_CSV = """ProviderId,ProductCategory,Amount,Value,Label
1,CategoryA,100,200,0
2,CategoryB,150,250,1
3,CategoryA,120,220,0
4,CategoryB,160,260,1
5,CategoryA,110,210,0
6,CategoryB,170,270,1
7,CategoryA,130,230,0
8,CategoryB,180,280,1
9,CategoryA,140,240,0
10,CategoryB,190,290,1
11,CategoryA,115,215,0
12,CategoryB,175,275,1
"""

# Mock feature names
MOCK_FEATURE_NAMES = ['ProviderId', 'ProductCategory', 'Amount', 'Value']

@pytest.fixture
def setup_logging():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

@pytest.fixture
def mock_models(tmp_dir):
    # Create mock models
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression())
    ])
    rf_model = RandomForestClassifier()
    
    # Save mock models
    model_dir = tmp_dir / "models"
    model_dir.mkdir()
    joblib.dump(lr_pipeline, model_dir / "LogisticRegression_best_model.pkl")
    joblib.dump(rf_model, model_dir / "RandomForest_best_model.pkl")
    
    # Mock feature names for RandomForest
    rf_model.feature_names_in_ = np.array(MOCK_FEATURE_NAMES)
    return model_dir

@pytest.fixture
def sample_data(tmp_dir):
    data_file = tmp_dir / "sample_data.csv"
    data_file.write_text(SAMPLE_CSV)
    return data_file

def test_load_models_success(mock_models, setup_logging):
    models = load_models(str(mock_models))
    assert models is not None
    assert 'LogisticRegression' in models
    assert 'RandomForest' in models
    assert setup_logging.handlers

def test_load_models_failure(tmp_dir, setup_logging):
    models = load_models(str(tmp_dir / "nonexistent"))
    assert models is None  # Expect None on failure instead of raising exception

def test_load_or_preprocess_data_success(sample_data, tmp_dir):
    X_train, X_test, y_train, y_test = load_or_preprocess_data(str(sample_data))
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert X_train.shape[1] == len(MOCK_FEATURE_NAMES)
    assert len(X_train) + len(X_test) == 12  # Updated to match new sample data

def test_load_or_preprocess_data_file_not_found(tmp_dir):
    result = load_or_preprocess_data(str(tmp_dir / "nonexistent.csv"))
    assert result is None  # Expect None on file not found instead of raising exception

def test_load_or_preprocess_data_invalid_csv(tmp_dir):
    invalid_file = tmp_dir / "invalid.csv"
    invalid_file.write_text('ProviderId,Amount\n"unclosed quote,100\n')
    result = load_or_preprocess_data(str(invalid_file))
    assert result is None  # Expect None on invalid CSV instead of raising exception

@patch('shap.TreeExplainer')
@patch('shap.plots.force')
@patch('shap.summary_plot')
@patch('shap.dependence_plot')
def test_explain_random_forest_success(
    mock_dependence_plot, mock_summary_plot, mock_force_plot, mock_tree_explainer,
    mock_models, sample_data, tmp_dir
):
    X_train, X_test, _, _ = load_or_preprocess_data(str(sample_data))
    rf_model = joblib.load(mock_models / "RandomForest_best_model.pkl")

    mock_explainer = Mock()
    mock_explainer.expected_value = [0.5, 0.6]
    mock_explainer.shap_values.return_value = np.zeros((X_test.shape[0], X_test.shape[1], 2))
    mock_tree_explainer.return_value = mock_explainer

    result = explain_random_forest(rf_model, X_train, X_test, MOCK_FEATURE_NAMES, str(tmp_dir / "explanations"))
    assert result is not None
    assert 'feature_importance' in result
    assert 'shap_values' in result
    assert isinstance(result['feature_importance'], pd.DataFrame)
    assert result['shap_values'].shape == (X_test.shape[0], X_test.shape[1])

def test_explain_logistic_regression_success(mock_models, sample_data, tmp_dir):
    X_train, _, _, _ = load_or_preprocess_data(str(sample_data))
    lr_model = joblib.load(mock_models / "LogisticRegression_best_model.pkl")
    # Preprocess X_train to handle categorical data and ensure numeric
    X_train_numeric = X_train[MOCK_FEATURE_NAMES].copy()
    X_train_numeric['ProductCategory'] = pd.to_numeric(X_train_numeric['ProductCategory'], errors='coerce').fillna(0)
    lr_model.named_steps['scaler'].fit(X_train_numeric)

    result = explain_logistic_regression(lr_model, X_train_numeric, MOCK_FEATURE_NAMES, str(tmp_dir / "explanations"))
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(MOCK_FEATURE_NAMES)

def test_run_explainability_pipeline_success(mock_models, sample_data, tmp_dir):
    result = run_explainability_pipeline(str(sample_data), str(mock_models), str(tmp_dir / "explanations"))
    assert result is not None
    assert 'LogisticRegression' in result
    assert 'RandomForest' in result
    assert isinstance(result['LogisticRegression'], pd.DataFrame)
    assert isinstance(result['RandomForest'], dict)

def test_run_explainability_pipeline_failure_model_load(tmp_dir, sample_data):
    result = run_explainability_pipeline(str(sample_data), str(tmp_dir / "nonexistent"), str(tmp_dir / "explanations"))
    assert result is None  # Expect None on model load failure

def test_run_explainability_pipeline_failure_data_load(tmp_dir, mock_models):
    result = run_explainability_pipeline(str(tmp_dir / "nonexistent.csv"), str(mock_models), str(tmp_dir / "explanations"))
    assert result is None  # Expect None on data load failure

if __name__ == "__main__":
    pytest.main([__file__])