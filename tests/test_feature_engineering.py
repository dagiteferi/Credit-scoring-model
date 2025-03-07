import pandas as pd
import numpy as np
import os
import pytest
from scripts.Feature_Engineering import (
    calculate_woe, create_aggregate_features, extract_time_features,
    encode_categorical_variables, check_and_handle_missing_values,
    standardize_numerical_features, construct_rfms_scores, save_transformed_data
)

# --- Tests for calculate_woe ---
def test_calculate_woe_standard():
    """Test WoE and IV calculation with a typical dataset."""
    df = pd.DataFrame({
        'feature': [1, 2, 3, 1, 2, 3],
        'Label': [0, 1, 0, 1, 0, 1]
    })
    woe_df, iv = calculate_woe(df, 'Label', 'feature')
    assert isinstance(woe_df, pd.DataFrame), "WoE output should be a DataFrame"
    assert 'woe' in woe_df.columns, "WoE column missing in woe_df"
    assert iv >= 0, "IV should be non-negative"
    assert 'feature_WOE' in df.columns, "WoE feature column not added to input DataFrame"

def test_calculate_woe_single_value():
    """Test WoE and IV when the feature has a single unique value."""
    df = pd.DataFrame({
        'feature': [1, 1, 1],
        'Label': [0, 1, 0]
    })
    woe_df, iv = calculate_woe(df, 'Label', 'feature')
    assert woe_df['woe'].iloc[0] == 0, "WoE should be 0 for single-value feature"
    assert iv == 0, "IV should be 0 for single-value feature"

def test_calculate_woe_missing_column():
    """Test that a ValueError is raised when required columns are missing."""
    df = pd.DataFrame({'feature': [1, 2, 3]})
    with pytest.raises(ValueError, match="Columns 'Label' or 'feature' not found"):
        calculate_woe(df, 'Label', 'feature')

# --- Tests for create_aggregate_features ---
def test_create_aggregate_features_standard():
    """Test aggregate feature creation with multiple transactions per customer."""
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300],
        'TransactionId': ['T1', 'T2', 'T3']
    })
    result = create_aggregate_features(df)
    assert 'Total_Transaction_Amount' in result.columns, "Total amount column missing"
    assert result['Total_Transaction_Amount'].iloc[0] == 300, "Incorrect total amount for customer 1"
    assert result['Transaction_Count'].iloc[0] == 2, "Incorrect transaction count for customer 1"
    # Fixed to use ddof=1 for sample standard deviation
    assert np.isclose(result['Std_Transaction_Amount'].iloc[0], np.std([100, 200], ddof=1)), "Incorrect std for customer 1"

def test_create_aggregate_features_single_transaction():
    """Test aggregate features when each customer has one transaction."""
    df = pd.DataFrame({
        'CustomerId': [1],
        'Amount': [100],
        'TransactionId': ['T1']
    })
    result = create_aggregate_features(df)
    assert result['Std_Transaction_Amount'].iloc[0] == 0, "Std should be 0 for single transaction"

def test_create_aggregate_features_missing_column():
    """Test that a ValueError is raised when required columns are missing."""
    df = pd.DataFrame({'CustomerId': [1, 2]})
    with pytest.raises(ValueError, match="Required columns 'Amount' or 'CustomerId' not found"):
        create_aggregate_features(df)

# --- Tests for extract_time_features ---
def test_extract_time_features_standard():
    """Test extraction of time features from TransactionStartTime."""
    df = pd.DataFrame({
        'TransactionStartTime': ['2023-01-01 12:00:00', '2023-01-02 13:00:00']
    })
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    result = extract_time_features(df)
    assert 'Transaction_Hour' in result.columns, "Hour column missing"
    assert result['Transaction_Hour'].iloc[0] == 12, "Incorrect hour extracted"
    assert result['Transaction_Day'].iloc[1] == 2, "Incorrect day extracted"

def test_extract_time_features_missing_column():
    """Test that a ValueError is raised when TransactionStartTime is missing."""
    df = pd.DataFrame({'Amount': [100, 200]})
    with pytest.raises(ValueError, match="Required column 'TransactionStartTime' not found"):
        extract_time_features(df)

# --- Tests for encode_categorical_variables ---
def test_encode_categorical_variables_standard():
    """Test encoding of categorical variables with WoE and one-hot encoding."""
    df = pd.DataFrame({
        'ProviderId': ['A', 'B', 'A'],
        'ChannelId': ['C1', 'C2', 'C1'],
        'Label': [0, 1, 0]
    })
    result = encode_categorical_variables(df, 'Label')
    assert 'ProviderId_WOE' in result.columns, "WoE column for ProviderId missing"
    assert 'ChannelId_C2' in result.columns, "One-hot encoded column missing"

def test_encode_categorical_variables_missing_target():
    """Test that a ValueError is raised when the target variable is missing."""
    df = pd.DataFrame({'ProviderId': ['A', 'B']})
    with pytest.raises(ValueError, match="Target variable 'Label' not found"):
        encode_categorical_variables(df, 'Label')

# --- Tests for check_and_handle_missing_values ---
def test_check_and_handle_missing_values_standard():
    """Test imputation of missing values with median."""
    df = pd.DataFrame({
        'Amount': [100, np.nan, 200],
        'Value': [10, 20, np.nan]
    })
    result = check_and_handle_missing_values(df)
    assert result.isnull().sum().sum() == 0, "Missing values not imputed"
    assert result['Amount'].iloc[1] == 150, "Incorrect imputation for Amount (median of 100, 200)"

def test_check_and_handle_missing_values_no_missing():
    """Test that the DataFrame remains unchanged when no missing values exist."""
    df = pd.DataFrame({
        'Amount': [100, 200],
        'Value': [10, 20]
    })
    result = check_and_handle_missing_values(df)
    assert result.equals(df), "DataFrame altered when no missing values present"

# --- Tests for standardize_numerical_features ---
def test_standardize_numerical_features_standard():
    """Test standardization of numerical features to mean 0 and std 1."""
    df = pd.DataFrame({
        'Amount': [100, 200, 300],
        'Total_Transaction_Amount': [400, 500, 600]
    })
    result = standardize_numerical_features(df)
    assert np.isclose(result['Amount'].mean(), 0, atol=1e-8), "Mean not zero after standardization"
    # Fixed to use ddof=0 for population standard deviation after standardization
    assert np.isclose(result['Amount'].std(ddof=0), 1, atol=1e-8), "Population std not one after standardization"

def test_standardize_numerical_features_no_numerical():
    """Test that the DataFrame remains unchanged when no numerical features exist."""
    df = pd.DataFrame({'Category': ['A', 'B']})
    result = standardize_numerical_features(df)
    assert result.equals(df), "DataFrame altered when no numerical features present"

# --- Tests for construct_rfms_scores ---
def test_construct_rfms_scores_standard():
    """Test RFMS score and label construction with typical data."""
    df = pd.DataFrame({
        'TransactionStartTime': [pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)] * 2,
        'Transaction_Count': [5, 10],
        'Total_Transaction_Amount': [400, 600]
    })
    result = construct_rfms_scores(df)
    assert 'RFMS_score' in result.columns, "RFMS score column missing"
    assert 'Label' in result.columns, "Label column missing"
    assert 'RFMS_score_binned_WOE' in result.columns, "WoE for binned RFMS missing"
    assert set(result['Label'].unique()) <= {0, 1}, "Labels should be 0 or 1"

def test_construct_rfms_scores_missing_column():
    """Test that a ValueError is raised when required columns are missing."""
    df = pd.DataFrame({'Transaction_Count': [5]})
    with pytest.raises(ValueError, match="Required columns missing"):
        construct_rfms_scores(df)

# --- Tests for save_transformed_data ---
def test_save_transformed_data_standard(tmp_path):
    """Test saving a DataFrame to a CSV file using a temporary path."""
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    output_path = tmp_path / "test_data.csv"
    save_transformed_data(df, str(output_path))
    assert output_path.exists(), "Output file not created"
    loaded_df = pd.read_csv(output_path)
    assert loaded_df.equals(df), "Saved data does not match original"