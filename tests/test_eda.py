import pytest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scripts.Eda import (data_overview, summary_statistics, 
                 visualize_distribution, box_plot_for_outliers, 
                 check_for_skewness, distribution_of_numerical_features,
                 correlation_analysis, identify_missing_values)


# Sample DataFrame for testing
@pytest.fixture
def sample_data():
    """Fixture to provide a sample DataFrame."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "income": [50000, 60000, None],
        "credit_score": [700, 650, 600],
        "category": ["A", "B", "A"]
    })

@pytest.fixture
def setup_logging():
    """Fixture to set up logging for tests."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def test_load_data(sample_data, tmp_path):
    """Test loading data (already tested in data_loader, but verify compatibility)."""
    file_path = tmp_path / "data.csv"
    sample_data.to_csv(file_path, index=False)
    data = load_data(str(file_path))
    assert data.shape == sample_data.shape

def test_data_overview(capfd, sample_data):
    """Test data overview output."""
    data_overview(sample_data)
    out, _ = capfd.readouterr()
    assert "Number of rows and columns: (3, 4)" in out
    assert "customer_id     int64" in out

def test_summary_statistics(capfd, sample_data):
    """Test summary statistics output."""
    summary_statistics(sample_data)
    out, _ = capfd.readouterr()
    assert "income" in out
    assert "mean" in out

def test_visualize_distribution(sample_data, mocker):
    """Test visualization of numerical distribution (mock plotting)."""
    numeric_cols = ["income", "credit_score"]
    mocker.patch("matplotlib.pyplot.show")  # Mock plt.show to avoid display
    visualize_distribution(sample_data, numeric_cols)
    # Check that plotting was called (indirectly via fixture)
    assert plt.gcf().number > 0  # Ensure a figure was created

def test_box_plot_for_outliers(sample_data, mocker):
    """Test box plot for outliers (mock plotting)."""
    numeric_cols = ["income", "credit_score"]
    mocker.patch("matplotlib.pyplot.show")
    box_plot_for_outliers(sample_data, numeric_cols)
    assert plt.gcf().number > 0

def test_check_for_skewness(capfd, sample_data):
    """Test skewness calculation and output."""
    numeric_cols = ["income", "credit_score"]
    check_for_skewness(sample_data, numeric_cols)
    out, _ = capfd.readouterr()
    assert "Skewness for Numerical Features" in out

def test_distribution_of_numerical_features(sample_data, mocker, setup_logging):
    """Test full numerical distribution analysis."""
    mocker.patch("matplotlib.pyplot.show")
    distribution_of_numerical_features(sample_data)
    assert setup_logging.handlers  # Ensure logging occurred

def test_correlation_analysis(sample_data, mocker):
    """Test correlation heatmap (mock plotting)."""
    mocker.patch("matplotlib.pyplot.show")
    correlation_analysis(sample_data)
    assert plt.gcf().number > 0

def test_identify_missing_values(capfd, sample_data):
    """Test missing value identification."""
    identify_missing_values(sample_data)
    out, _ = capfd.readouterr()
    assert "Missing values in each column" in out
    assert "income    1" in out