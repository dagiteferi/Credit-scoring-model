import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scripts.Eda import (data_overview, summary_statistics, 
                        visualize_distribution, box_plot_for_outliers, 
                        check_for_skewness, distribution_of_numerical_features,
                        correlation_analysis, identify_missing_values)

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "income": [50000, 60000, None],
        "credit_score": [700, 650, 600],  # Fixed '600SMALL' to 600
        "category": ["A", "B", "A"]
    })

@pytest.fixture
def setup_logging():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def test_load_data(sample_data, tmp_path):
    file_path = tmp_path / "data.csv"
    sample_data.to_csv(file_path, index=False)
    data = load_data(str(file_path))
    assert data.shape == sample_data.shape

def test_data_overview(capfd, sample_data):
    data_overview(sample_data)
    out, _ = capfd.readouterr()
    assert "Number of rows and columns: (3, 4)" in out
    assert "customer_id" in out and "int64" in out

def test_summary_statistics(capfd, sample_data):
    summary_statistics(sample_data)
    out, _ = capfd.readouterr()
    assert "income" in out
    assert "mean" in out

def test_visualize_distribution(sample_data, monkeypatch):
    numeric_cols = ["income", "credit_score"]
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    visualize_distribution(sample_data, numeric_cols)
    assert plt.gcf().number > 0

def test_box_plot_for_outliers(sample_data, monkeypatch):
    numeric_cols = ["income", "credit_score"]
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    box_plot_for_outliers(sample_data, numeric_cols)
    assert plt.gcf().number > 0

def test_check_for_skewness(capfd, sample_data):
    numeric_cols = ["income", "credit_score"]
    check_for_skewness(sample_data, numeric_cols)
    out, _ = capfd.readouterr()
    assert "Skewness for Numerical Features" in out

def test_distribution_of_numerical_features(sample_data, monkeypatch, setup_logging):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    distribution_of_numerical_features(sample_data)
    assert setup_logging.handlers

def test_correlation_analysis(sample_data, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    correlation_analysis(sample_data)
    assert plt.gcf().number > 0

def test_identify_missing_values(capfd, sample_data):
    identify_missing_values(sample_data)
    out, _ = capfd.readouterr()
    assert "Missing values in each column" in out
    assert "income    1" in out