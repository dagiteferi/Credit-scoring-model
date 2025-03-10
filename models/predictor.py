# models/predictor.py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from credit_scoring_app.config import logger

def load_model(model_path: str):
    """Load the trained machine learning model from the specified path."""
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def preprocess_data(data: dict) -> pd.DataFrame:
    """Preprocess raw input data to match the features expected by the model."""
    try:
        # Convert the input dictionary to a DataFrame
        df = pd.DataFrame([data])

        # Extract features from TransactionStartTime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
        df['Transaction_Day'] = df['TransactionStartTime'].dt.day
        df['Transaction_Month'] = df['TransactionStartTime'].dt.month
        df['Transaction_Year'] = df['TransactionStartTime'].dt.year
        # Duplicate features as TransactionHour, TransactionDay, TransactionMonth (as per model)
        df['TransactionHour'] = df['Transaction_Hour']
        df['TransactionDay'] = df['Transaction_Day']
        df['TransactionMonth'] = df['Transaction_Month']

        # Simulate feature engineering (align with training data)
        df['Total_Transaction_Amount'] = df['Amount']
        df['Average_Transaction_Amount'] = df['Amount']
        df['Transaction_Count'] = 1
        df['Std_Transaction_Amount'] = 0
        df['Recency'] = (datetime.now() - df['TransactionStartTime']).dt.days
        df['RFMS_score'] = 0.5  # Placeholder
        df['RFMS_score_binned'] = 0  # Placeholder (adjust binning logic if known)

        # Add missing features with default values
        df['ProviderId'] = 0  # Default value (adjust if needed)
        df['ProductCategory'] = 'Unknown'  # Default value (adjust if needed)
        df['Value'] = 0  # Default value (adjust if needed)
        df['PricingStrategy'] = 0  # Default value (adjust if needed)

        # WOE features (placeholders, adjust if actual WOE logic is known)
        df['RFMS_score_binned_WOE'] = 0.0
        df['ProviderId_WOE'] = 0.0
        df['ProviderId_WOE.1'] = 0.0
        df['ProductId_WOE'] = 0.0
        df['ProductId_WOE.1'] = 0.0
        df['ProductCategory_WOE'] = 0.0
        df['ProductCategory_WOE.1'] = 0.0

        # One-hot encode ChannelId with the correct prefix (ChannelId_ChannelId_X)
        df = pd.get_dummies(df, columns=['ChannelId'], prefix='ChannelId_ChannelId')
        # Ensure all possible ChannelId values from training are included
        for value in [2, 3, 5]:  # From training data
            col_name = f'ChannelId_ChannelId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        # One-hot encode ProductId with the correct prefix (ProductId_X)
        df = pd.get_dummies(df, columns=['ProductId'], prefix='ProductId')
        # Ensure all possible ProductId values from training are included
        for value in range(1, 23):  # ProductId_1 to ProductId_22
            col_name = f'ProductId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        # Drop raw columns that are not used by the model
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                           'CurrencyCode', 'CountryCode', 'TransactionStartTime']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Define the exact features expected by the model (from model.feature_names_in_)
        expected_features = [
            'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy',
            'Transaction_Count', 'Std_Transaction_Amount', 'Transaction_Hour',
            'Transaction_Day', 'Transaction_Month', 'Transaction_Year', 'Recency',
            'RFMS_score', 'RFMS_score_binned', 'RFMS_score_binned_WOE', 'ProviderId_WOE',
            'ProviderId_WOE.1', 'ProductId_WOE', 'ProductId_WOE.1', 'ProductCategory_WOE',
            'ProductCategory_WOE.1', 'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3',
            'ChannelId_ChannelId_5', 'ProductId_1', 'ProductId_2', 'ProductId_3',
            'ProductId_4', 'ProductId_5', 'ProductId_6', 'ProductId_7', 'ProductId_8',
            'ProductId_9', 'ProductId_10', 'ProductId_11', 'ProductId_12', 'ProductId_13',
            'ProductId_14', 'ProductId_15', 'ProductId_16', 'ProductId_17', 'ProductId_18',
            'ProductId_19', 'ProductId_20', 'ProductId_21', 'ProductId_22',
            'TransactionHour', 'TransactionDay', 'TransactionMonth'
        ]

        # Add missing columns with default values (0)
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Ensure the DataFrame has exactly the expected features in the correct order
        df = df[expected_features]

        logger.info("Data preprocessed successfully")
        return df
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def predict(model, data: dict) -> int:
    """Make a prediction using the loaded model and preprocessed data."""
    try:
        # Preprocess the input data
        X = preprocess_data(data)

        # Make prediction
        prediction = model.predict(X)[0]
        logger.info(f"Prediction made: {prediction}")
        return int(prediction)  # Ensure the prediction is returned as an integer
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise