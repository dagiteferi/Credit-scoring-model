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
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Custom WOE implementation
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
        woe_df = df.groupby(feature).agg(
            count=(target, 'size'),
            event=(target, 'sum')
        ).reset_index()

        woe_df['non_event'] = woe_df['count'] - woe_df['event']
        total_events = woe_df['event'].sum()
        total_non_events = woe_df['non_event'].sum()

        # Avoid division by zero
        woe_df['event_rate'] = np.where(total_events == 0, 0, woe_df['event'] / total_events)
        woe_df['non_event_rate'] = np.where(total_non_events == 0, 0, woe_df['non_event'] / total_non_events)

        # Calculate WOE with handling for zero rates
        woe_df['woe'] = np.where(
            (woe_df['event_rate'] == 0) | (woe_df['non_event_rate'] == 0),
            0,  # Default value for undefined WOE
            np.log(woe_df['event_rate'] / woe_df['non_event_rate']).replace([-np.inf, np.inf], 0)
        )

        logger.info(f"WOE calculation completed for {feature}")
        return woe_df[[feature, 'count', 'event', 'non_event', 'woe']]

    except ValueError as ve:
        logger.error(f"ValueError in WOE calculation: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in WOE calculation: {e}")
        return None

class WOE:
    def __init__(self):
        self.woe_dict = {}
        self.iv_values_ = {}

    def fit(self, X, y):
        """
        Fit the WOE transformation to the data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target variable.
        """
        logger.info("Fitting WOE transformation")
        self.woe_dict = {}
        self.iv_values_ = {}
        for column in X.columns:
            self.woe_dict[column] = {}
            total_good = (y == 0).sum()
            total_bad = (y == 1).sum()
            for value in X[column].unique():
                good = ((X[column] == value) & (y == 0)).sum()
                bad = ((X[column] == value) & (y == 1)).sum()
                woe = np.log((good / total_good) / (bad / total_bad)) if (bad > 0 and good > 0) else 0
                iv = ((good / total_good) - (bad / total_bad)) * woe if (bad > 0 and good > 0) else 0
                self.woe_dict[column][value] = woe
                self.iv_values_.setdefault(column, 0)
                self.iv_values_[column] += iv
        logger.info(f"WOE fit completed with IV values: {self.iv_values_}")

    def transform(self, X):
        """
        Transform the data using fitted WOE values.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame with WOE values.
        """
        logger.info("Transforming data with WOE")
        X_woe = X.copy()
        for column in X.columns:
            X_woe[column] = X_woe[column].map(self.woe_dict[column]).fillna(0)
        logger.info("WOE transformation completed")
        return X_woe

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file is not found.
        Exception: For other loading errors.
    """
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape: {data.shape}")
        return data
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

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
        aggregates = data.groupby('CustomerId').agg(
            Total_Transaction_Amount=('Amount', 'sum'),
            Average_Transaction_Amount=('Amount', 'mean'),
            Transaction_Count=('TransactionId', 'count'),
            Std_Transaction_Amount=('Amount', 'std')
        ).reset_index()
        data = data.merge(aggregates, on='CustomerId', how='left')
        logger.info("Aggregate features created successfully")
        return data
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
        data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
        data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
        data['Transaction_Day'] = data['TransactionStartTime'].dt.day
        data['Transaction_Month'] = data['TransactionStartTime'].dt.month
        data['Transaction_Year'] = data['TransactionStartTime'].dt.year
        logger.info("Time features extracted successfully")
        return data
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
        # Convert categorical columns to category type for WOE
        categorical_cols = ['CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory']
        for col in categorical_cols:
            if data[col].dtype == 'object':
                data[col] = data[col].astype('category').cat.codes
                logger.info(f"Converted {col} to category codes")

        # Apply WOE encoding
        logger.info("Applying WOE encoding")
        woe_encoder = WOE()
        woe_encoder.fit(data[categorical_cols], data[target_variable])
        data_woe = woe_encoder.transform(data[categorical_cols])
        data_woe.columns = [f"{col}_WOE" for col in data_woe.columns]
        data = pd.concat([data, data_woe], axis=1)
        logger.info(f"WOE IV values: {woe_encoder.iv_values_}")

        # Apply One-Hot Encoding for nominal variables
        logger.info("Applying One-Hot Encoding")
        data = pd.get_dummies(data, columns=['ChannelId'], drop_first=True)
        logger.info("Categorical encoding completed")
        return data

    except Exception as e:
        logger.error(f"Error encoding categorical variables: {e}")
        return data

def check_and_handle_missing_values(data):
    """
    Check and handle missing values using imputation or removal.

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

def normalize_standardize_features(data):
    """
    Normalize and standardize numerical features.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with normalized/standardized features.
    """
    logger.info("Normalizing and standardizing numerical features")
    try:
        numerical_features = ['Amount', 'Total_Transaction_Amount', 'Average_Transaction_Amount',
                             'Transaction_Count', 'Std_Transaction_Amount']
        # Standardize features
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
        logger.info(f"Standardized features sample:\n{data[numerical_features].head()}")
        return data
    except Exception as e:
        logger.error(f"Error normalizing/standardizing features: {e}")
        return data

def construct_rfms_scores(data):
    """
    Construct RFMS (Recency, Frequency, Monetary, Score) features and assign labels.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Dataset with RFMS scores and labels.
    """
    logger.info("Constructing RFMS scores")
    try:
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
        plt.figure(figsize=(10, 6))
        plt.scatter(data['Transaction_Count'], data['Total_Transaction_Amount'], c=data['RFMS_score'], cmap='viridis')
        plt.colorbar(label='RFMS Score')
        plt.xlabel('Transaction Count')
        plt.ylabel('Total Transaction Amount')
        plt.title('RFMS Visualization')
        plt.show()

        # Calculate WOE for RFMS_score
        woe_results = calculate_woe(data, 'Label', 'RFMS_score')
        if woe_results is not None:
            print("WOE Results for RFMS_score:\n", woe_results)
            logger.info("WOE calculation for RFMS_score completed")
        return data

    except Exception as e:
        logger.error(f"Error constructing RFMS scores: {e}")
        return data

