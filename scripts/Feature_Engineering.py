# feature_engineering.py
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
        target (str): Target column name (e.g., 'FraudResult').
        feature (str): Feature column name to calculate WOE for.

    Returns:
        pd.DataFrame: DataFrame with WOE values, counts, events, and non-events.

    Raises:
        ValueError: If target or feature column is not found.
        Exception: For other unexpected errors.
    """
    try:
        logger.info(f"Calculating WOE for feature: {feature}")
        if target not in df.columns or feature not in df.columns:
            raise ValueError(f"Columns '{target}' or '{feature}' not found in DataFrame")

        # Convert target to numeric and handle errors
        df[target] = pd.to_numeric(df[target], errors='coerce')
        woe_dict = {}
        iv_values = {}

        # Aggregate data for WOE calculation
        grouped = df.groupby(feature)
        total_good = (df[target] == 0).sum()
        total_bad = (df[target] == 1).sum()

        # Handle case where total_good or total_bad is zero
        if total_good == 0 or total_bad == 0:
            logger.warning(f"Total good or bad counts are zero for {target}. Setting WOE to 0.")
            woe_dict = {value: 0 for value in grouped[feature].unique()}
            iv_values[feature] = 0
        else:
            for value in grouped[feature].unique():
                mask = df[feature] == value
                good = (mask & (df[target] == 0)).sum()
                bad = (mask & (df[target] == 1)).sum()
                woe = np.log((good / total_good) / (bad / total_bad)) if (bad > 0 and good > 0) else 0
                iv = ((good / total_good) - (bad / total_bad)) * woe if (bad > 0 and good > 0) else 0
                woe_dict[value] = woe
                iv_values[feature] = iv_values.get(feature, 0) + iv

        # Transform the feature with WOE values
        df[feature + '_WOE'] = df[feature].map(woe_dict).fillna(0)
        logger.info(f"WOE and IV calculated for {feature}. IV: {iv_values}")

        # Return WOE summary
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
        return woe_df[[feature, 'count', 'event', 'non_event', 'woe']]

    except ValueError as ve:
        logger.error(f"ValueError in WOE calculation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in WOE calculation: {e}")
        return None

def create_aggregate_features(data):
    """
    Create aggregate features for each customer (Total, Average, Count, Std of Transaction Amounts).

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with aggregated features.
    """
    logger.info("Creating aggregate features for customers")
    try:
        # Filter out negative amounts to avoid skewing aggregates
        if 'Amount' not in data.columns or 'CustomerId' not in data.columns:
            raise ValueError("Required columns 'Amount' or 'CustomerId' not found in DataFrame")

        filtered_data = data[data['Amount'] > 0]
        logger.info("Filtered data for positive transactions")
        aggregates = filtered_data.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()
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

def encode_categorical_variables(data, target_variable='FraudResult'):
    """
    Encode categorical variables using WOE, Label Encoding, and One-Hot Encoding.

    Args:
        data (pd.DataFrame): Input dataset.
        target_variable (str): Target column for WOE calculation.

    Returns:
        pd.DataFrame: Dataset with encoded categorical variables.
    """
    logger.info("Encoding categorical variables")
    try:
        if target_variable not in data.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in DataFrame")

        # Define categorical columns for WOE encoding
        categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']
        for col in categorical_cols:
            if col in data.columns and data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes
                logger.info(f"Converted {col} to category codes")

        # Apply WOE encoding
        logger.info("Applying WOE encoding")
        for col in categorical_cols:
            if col in data.columns:
                woe_df = calculate_woe(data, target_variable, col)
                if woe_df is not None:
                    data = pd.concat([data, pd.DataFrame({f"{col}_WOE": data[col].map(dict(zip(woe_df[col], woe_df['woe']))).fillna(0)})], axis=1)

        # Apply One-Hot Encoding for nominal variables
        logger.info("Applying One-Hot Encoding")
        if 'ChannelId' in data.columns:
            data = pd.get_dummies(data, columns=['ChannelId'], drop_first=True)
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
    """
    logger.info("Standardizing numerical features")
    try:
        numerical_features = ['Amount', 'Value', 'Total_Transaction_Amount', 'Average_Transaction_Amount',
                             'Transaction_Count', 'Std_Transaction_Amount']
        # Ensure all features exist in the dataset
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

def construct_rfms_scores(data):
    """
    Construct RFMS (Recency, Frequency, Monetary, Score) features, assign labels, and perform WoE binning.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with RFMS scores, labels, and WoE binning.
    """
    logger.info("Constructing RFMS scores")
    try:
        # Validate required columns
        required_cols = ['TransactionStartTime', 'Transaction_Count', 'Total_Transaction_Amount']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Required columns missing: {missing_cols}")

        # Convert to datetime
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        current_date = dt.datetime.now(dt.timezone.utc)
        data['Recency'] = (current_date - data['TransactionStartTime']).dt.days

        # Calculate RFMS score
        data['RFMS_score'] = (1 / (data['Recency'] + 1) * 0.4) + (data['Transaction_Count'] * 0.3) + \
                           (data['Total_Transaction_Amount'] * 0.3)
        data['RFMS_score'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['RFMS_score'].fillna(0, inplace=True)

        # Assign labels based on median threshold
        threshold = data['RFMS_score'].median()
        data['Label'] = np.where(data['RFMS_score'] > threshold, 1, 0)
        logger.info(f"RFMS score distribution:\n{data['Label'].value_counts()}")

        # Visualize RFMS space
        logger.info("Visualizing RFMS space")
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Transaction_Count'], data['Total_Transaction_Amount'], c=data['RFMS_score'], cmap='viridis')
        plt.colorbar(label='RFMS Score')
        plt.xlabel('Transaction Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('RFMS Visualization')
        plt.show()

        # Calculate WOE for RFMS_score
        logger.info("Calculating WOE for RFMS_score")
        woe_results = calculate_woe(data, 'Label', 'RFMS_score')
        if woe_results is not None:
            print("WOE Results for RFMS_score:\n", woe_results)
            logger.info("WOE calculation for RFMS_score completed")
        return data

    except ValueError as ve:
        logger.error(f"ValueError in constructing RFMS scores: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error constructing RFMS scores: {e}")
        return data

