# models/predictor.py
import joblib
import numpy as np
import pandas as pd
from config import logger

def load_model(path):
    """Load the trained model from the specified path."""
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_data(data):
    """
    Preprocess the dataset by checking for duplicate columns, handling missing values, encoding categorical variables,
    and extracting datetime features.

    Parameters:
    - data (dict): Raw input data as a dictionary.

    Returns:
    - pd.DataFrame: Preprocessed dataset, or raises an exception if an error occurs.
    """
    logger.info("Starting data preprocessing")
    try:
        # Convert dict to DataFrame
        df = pd.DataFrame([data])

        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()]
        num_duplicates = len(duplicate_columns)
        if num_duplicates > 0:
            logger.warning(f"Found {num_duplicates} duplicate columns: {list(duplicate_columns)}. Removing duplicates.")
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            logger.info("No duplicate columns found.")

        # Drop unnecessary columns if they exist
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            logger.info(f"Dropped unnecessary columns: {existing_columns_to_drop}")
        else:
            logger.info("No unnecessary columns to drop.")

        # Handle missing values
        missing_values = df.isnull().sum()
        missing_columns = missing_values[missing_values > 0]
        if not missing_columns.empty:
            logger.info("Missing values found in the following columns:")
            for col, count in missing_columns.items():
                logger.info(f"  - {col}: {count} missing values")
            total_missing = missing_columns.sum()
            for col in df.select_dtypes(include=['number']).columns:
                df[col] = df[col].fillna(df[col].median())
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna(df[col].mode()[0])
            logger.info(f"Handled {total_missing} missing values in total.")
        else:
            logger.info("No missing values found.")

        # Encode categorical variables
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
        existing_categorical_columns = [col for col in categorical_columns if col in df.columns]
        if existing_categorical_columns:
            df = pd.get_dummies(df, columns=existing_categorical_columns, drop_first=True)
            logger.info(f"Encoded categorical variables: {existing_categorical_columns}")
        else:
            logger.info("No categorical columns to encode.")

        # Extract datetime features from TransactionStartTime
        if 'TransactionStartTime' in df.columns:
            logger.info("Extracting datetime features from 'TransactionStartTime'")
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
            df['TransactionHour'] = df['TransactionStartTime'].dt.hour
            df['TransactionDay'] = df['TransactionStartTime'].dt.day
            df['TransactionMonth'] = df['TransactionStartTime'].dt.month
            df['TransactionWeekday'] = df['TransactionStartTime'].dt.weekday
            df = df.drop(columns=['TransactionStartTime'])
            logger.info("Datetime features extracted: 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionWeekday'")
        else:
            logger.info("'TransactionStartTime' column not found. Skipping datetime feature extraction.")

        logger.info("Data preprocessing completed successfully")
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def predict(model, data):
    """Make a prediction using the loaded model and preprocessed input data."""
    processed_data = preprocess_data(data)
    input_array = processed_data.values
    prediction = model.predict(input_array)
    return float(prediction[0])  # Assuming binary output (0 or 1)