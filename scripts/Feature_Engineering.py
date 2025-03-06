import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import sys

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(__file__), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    logging.info(f"Created logs directory: {log_dir}")

# Configure logging to store in logs/feature_engineering.log
log_file_path = os.path.join(log_dir, "feature_engineering.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def calculate_woe(df, target, feature):
    """
    Calculate the Weight of Evidence (WoE) for a given feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target column name (e.g., 'Label').
        feature (str): Feature column name to calculate WOE for.

    Returns:
        tuple: (pd.DataFrame with WOE values, counts, events, and non-events, Information Value).

    Raises:
        ValueError: If target or feature column is not found.
        Exception: For other unexpected errors.
    """
    try:
        logger.info(f"Calculating WOE for feature: {feature}")
        if target not in df.columns or feature not in df.columns:
            raise ValueError(f"Columns '{target}' or '{feature}' not found in DataFrame")

        df[target] = pd.to_numeric(df[target], errors='coerce')
        woe_dict = {}
        iv_total = 0

        unique_values = df[feature].unique()
        total_good = (df[target] == 0).sum()
        total_bad = (df[target] == 1).sum()

        if total_good == 0 or total_bad == 0:
            logger.warning(f"Total good or bad counts are zero for {target}. Setting WOE to 0.")
            woe_dict = {value: 0 for value in unique_values}
            iv_total = 0
        else:
            for value in unique_values:
                if pd.isna(value):
                    mask = df[feature].isna()
                else:
                    mask = df[feature] == value
                good = (mask & (df[target] == 0)).sum()
                bad = (mask & (df[target] == 1)).sum()
                woe = np.log((good / total_good) / (bad / total_bad)) if (bad > 0 and good > 0) else 0
                iv = ((good / total_good) - (bad / total_bad)) * woe if (bad > 0 and good > 0) else 0
                woe_dict[value] = woe
                iv_total += iv

        df[feature + '_WOE'] = df[feature].map(woe_dict).fillna(0)
        logger.info(f"WOE and IV calculated for {feature}. IV: {iv_total}")

        woe_df = df.groupby(feature).agg(
            count=(target, 'size'),
            event=(target, 'sum')
        ).reset_index()
        woe_df['non_event'] = woe_df['count'] - woe_df['event']
        total_events = woe_df['event'].sum()
        total_non_events = woe_df['non_event'].sum()
        woe_df['event_rate'] = np.where(total_events == 0, 0, woe_df['event'] / total_events)
        woe_df['non_event_rate'] = np.where(total_non_events == 0, 0, woe_df['non_event'] / total_non_events)
        woe_df['woe'] = woe_df[feature].map(woe_dict).fillna(0)

        logger.info(f"WOE calculation completed for {feature}")
        return woe_df[[feature, 'count', 'event', 'non_event', 'woe']], iv_total
    except ValueError as ve:
        logger.error(f"ValueError in WOE calculation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in WOE calculation: {e}")
        return None, 0

def create_aggregate_features(data):
    """
    Create aggregate features for each customer (Total, Average, Count, Std of Transaction Amounts).

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with aggregated features.

    Raises:
        ValueError: If required columns 'Amount' or 'CustomerId' are not found.
        Exception: For other unexpected errors.
    """
    logger.info("Creating aggregate features for customers")
    try:
        # Validate required columns
        if 'Amount' not in data.columns or 'CustomerId' not in data.columns:
            raise ValueError("Required columns 'Amount' or 'CustomerId' not found in DataFrame")

        # Process all transactions (no filtering for positive amounts)
        logger.info("Processing all transactions")
        aggregates = data.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()

        # Impute Std_Transaction_Amount where it is NaN (e.g., single transaction)
        aggregates['Std_Transaction_Amount'] = aggregates['Std_Transaction_Amount'].fillna(0)

        # Merge aggregates back to the original data, preserving all rows
        data = data.merge(aggregates, on='CustomerId', how='left')
        logger.info("Aggregate features created successfully")
        return data
    except ValueError as ve:
        logger.error(f"ValueError in creating aggregate features: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error creating aggregate features: {e}")
        return data

def extract_time_features(data):
    """
    Extract time-based features from TransactionStartTime.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with extracted time features.

    Raises:
        ValueError: If required column 'TransactionStartTime' is not found.
        Exception: For other unexpected errors.
    """
    logger.info("Extracting time-based features")
    try:
        if 'TransactionStartTime' not in data.columns:
            raise ValueError("Required column 'TransactionStartTime' not found in DataFrame")

        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
        data['Transaction_Day'] = data['TransactionStartTime'].dt.day
        data['Transaction_Month'] = data['TransactionStartTime'].dt.month
        data['Transaction_Year'] = data['TransactionStartTime'].dt.year
        logger.info("Time features extracted successfully")
        return data
    except ValueError as ve:
        logger.error(f"ValueError in extracting time features: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error extracting time features: {e}")
        return data

def encode_categorical_variables(data, target_variable='Label'):
    """
    Encode categorical variables using WOE and One-Hot Encoding.

    Args:
        data (pd.DataFrame): Input dataset.
        target_variable (str): Target column for WOE calculation (default: 'Label').

    Returns:
        pd.DataFrame: Dataset with encoded categorical variables.

    Raises:
        ValueError: If target variable is not found.
        Exception: For other unexpected errors.
    """
    logger.info("Encoding categorical variables")
    try:
        if target_variable not in data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in DataFrame")

        # Define categorical columns for WOE encoding, excluding constant CurrencyCode
        categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory']
        for col in categorical_cols:
            if col in data.columns and data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes
                logger.info(f"Converted {col} to category codes")

        # Apply WOE encoding
        logger.info("Applying WOE encoding")
        iv_dict = {}
        for col in categorical_cols:
            if col in data.columns and f"{col}_WOE" not in data.columns:  # Avoid duplicate columns
                woe_df, iv = calculate_woe(data, target_variable, col)
                if woe_df is not None:
                    data = pd.concat([data, pd.DataFrame({f"{col}_WOE": data[col].map(dict(zip(woe_df[col], woe_df['woe']))).fillna(0)})], axis=1)
                    iv_dict[col] = iv

        # Apply One-Hot Encoding for ChannelId
        logger.info("Applying One-Hot Encoding")
        if 'ChannelId' in data.columns and any(f"ChannelId_ChannelId_{i}" not in data.columns for i in range(2, 6)):
            data = pd.get_dummies(data, columns=['ChannelId'], drop_first=True, prefix='ChannelId')
        logger.info(f"Information Values for features: {iv_dict}")
        logger.info("Categorical encoding completed")
        return data
    except ValueError as ve:
        logger.error(f"ValueError in encoding categorical variables: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error encoding categorical variables: {e}")
        return data

def check_and_handle_missing_values(data):
    """
    Check and handle missing values using imputation.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with handled missing values.

    Raises:
        Exception: For unexpected errors.
    """
    logger.info("Checking for missing values")
    try:
        missing_values = data.isnull().sum()
        logger.info(f"Missing values:\n{missing_values}")
        if missing_values.sum() > 0:
            logger.info("Imputing missing values with median")
            imputer = SimpleImputer(strategy='median')
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
            logger.info("Missing values imputed successfully")
        else:
            logger.info("No missing values found")
        return data
    except Exception as e:
        logger.error(f"Error handling missing values: {e}")
        return data

def standardize_numerical_features(data):
    """
    Standardize numerical features to have a mean of 0 and standard deviation of 1.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with standardized numerical features.

    Raises:
        Exception: For unexpected errors.
    """
    logger.info("Standardizing numerical features")
    try:
        numerical_features = ['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount',
                             'Transaction_Count', 'Std_Transaction_Amount']
        numerical_features = [col for col in numerical_features if col in data.columns]
        if not numerical_features:
            logger.warning("No numerical features found for standardization")
            return data

        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        logger.info(f"Standardized features sample:\n{data[numerical_features].head()}")
        return data
    except Exception as e:
        logger.error(f"Error standardizing numerical features: {e}")
        return data

def custom_bin_rfms(data, n_bins=10):
    logger.info(f"Custom binning RFMS_score into {n_bins} bins with class mixing")
    # Sort by RFMS_score and alternate between Label groups
    sorted_data = data.sort_values('RFMS_score')
    bin_edges = np.linspace(sorted_data.index[0], sorted_data.index[-1], n_bins + 1, dtype=int)
    data['RFMS_score_binned'] = pd.cut(sorted_data.index, bins=bin_edges, labels=False, include_lowest=True)
    return data

def construct_rfms_scores(data):
    logger.info("Constructing RFMS scores")
    try:
        required_cols = ['TransactionStartTime', 'Transaction_Count', 'Total_Transaction_Amount']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        current_date = pd.Timestamp.now(tz='UTC')
        data['Recency'] = (current_date - data['TransactionStartTime']).dt.days

        data['RFMS_score'] = (1 / (data['Recency'] + 1) * 0.4) + (data['Transaction_Count'] * 0.3) + \
                           (data['Total_Transaction_Amount'] * 0.3)
        data['RFMS_score'] = data['RFMS_score'].replace([np.inf, -np.inf], np.nan).fillna(0)

        threshold = data['RFMS_score'].median()
        data['Label'] = np.where(data['RFMS_score'] > threshold, 1, 0)
        logger.info(f"RFMS score distribution:\n{data['Label'].value_counts()}")

        logger.info("Visualizing RFMS space")
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Transaction_Count'], data['Total_Transaction_Amount'], c=data['RFMS_score'], cmap='viridis')
        plt.colorbar(label='RFMS Score')
        plt.xlabel('Transaction Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('RFMS Visualization')
        plt.show()

        logger.info("Binning RFMS_score for WOE calculation")
        # Use pandas.cut with 20 bins to avoid NaN from duplicates
        data['RFMS_score_binned'] = pd.cut(data['RFMS_score'], bins=20, labels=False, include_lowest=True)
        # Fill any remaining NaN with the median bin
        data['RFMS_score_binned'] = data['RFMS_score_binned'].fillna(data['RFMS_score_binned'].median())
        logger.info("Validating bin distribution")
        print("RFMS_score_binned distribution:\n", data['RFMS_score_binned'].value_counts())
        print("Label distribution by RFMS_score_binned:\n", data.groupby('RFMS_score_binned')['Label'].value_counts())

        logger.info("Calculating WOE for RFMS_score_binned")
        woe_results = calculate_woe(data, 'Label', 'RFMS_score_binned')
        if woe_results is not None:
            print("WOE Results for RFMS_score_binned:\n", woe_results[0])
            # Map WOE only for valid bins, set NaN to 0 if not matched
            woe_mapping = dict(zip(woe_results[0]['RFMS_score_binned'], woe_results[0]['woe']))
            data['RFMS_score_binned_WOE'] = data['RFMS_score_binned'].map(woe_mapping).fillna(0)
            logger.info("WOE calculation for RFMS_score completed")
        return data
    except ValueError as ve:
        logger.error(f"ValueError in constructing RFMS scores: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error constructing RFMS scores: {e}")
        return data
def save_transformed_data(data, output_path="data/transformed_data_credit_scoring.csv", file_format="csv"):
    logger.info(f"Saving transformed data to {output_path}")
    try:
        output_dir = os.path.dirname(output_path) or '.'
        if output_dir != '.':
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        if file_format == "csv":
            data.to_csv(output_path, index=False)
        elif file_format == "pickle":
            data.to_pickle(output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Successfully saved transformed data to {output_path}")
        logger.info(f"Total columns saved: {len(data.columns)}")
        logger.info(f"Column names saved: {data.columns.tolist()}")

    except OSError as e:
        logger.error(f"OSError while saving transformed data to {output_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving transformed data to {output_path}: {e}")
        raise