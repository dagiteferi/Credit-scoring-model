# models/predictor.py
import sys
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Add the root directory to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from credit_scoring_app.config
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

def preprocess_data(data: dict) -> tuple[pd.DataFrame, float]:
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

        # Calculate Recency
        current_date = pd.Timestamp.now(tz='UTC')
        df['Recency'] = (current_date - df['TransactionStartTime']).dt.days

        # Feature engineering
        df['Total_Transaction_Amount'] = df['Amount']
        df['Average_Transaction_Amount'] = df['Amount']
        df['Transaction_Count'] = 1
        df['Std_Transaction_Amount'] = 0

        # Calculate RFMS_score
        df['RFMS_score'] = (1 / (df['Recency'] + 1) * 0.4) + (df['Transaction_Count'] * 0.3) + (df['Total_Transaction_Amount'] * 0.3)
        df['RFMS_score'] = df['RFMS_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['RFMS_score_binned'] = 0  # Placeholder (adjust binning logic if known)

        # Add missing features with default values
        df['ProviderId'] = 0
        df['ProductCategory'] = 0
        df['Value'] = 0
        df['PricingStrategy'] = 0
        df['FraudResult'] = 0

        # WOE features (placeholders)
        df['RFMS_score_binned_WOE'] = 0.0
        df['ProviderId_WOE'] = 0.0
        df['ProviderId_WOE.1'] = 0.0
        df['ProductId_WOE'] = 0.0
        df['ProductId_WOE.1'] = 0.0
        df['ProductCategory_WOE'] = 0.0
        df['ProductCategory_WOE.1'] = 0.0

        # One-hot encode ChannelId
        df = pd.get_dummies(df, columns=['ChannelId'], prefix='ChannelId_ChannelId')
        for value in [2, 3, 5]:
            col_name = f'ChannelId_ChannelId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        # One-hot encode ProductId
        df = pd.get_dummies(df, columns=['ProductId'], prefix='ProductId')
        for value in range(1, 23):
            col_name = f'ProductId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        # Drop raw columns
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                           'CurrencyCode', 'CountryCode', 'TransactionStartTime']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Define expected features
        expected_features = [
            'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy',
            'FraudResult', 'Total_Transaction_Amount', 'Average_Transaction_Amount',
            'Transaction_Count', 'Std_Transaction_Amount', 'Transaction_Hour',
            'Transaction_Day', 'Transaction_Month', 'Transaction_Year', 'Recency',
            'RFMS_score', 'RFMS_score_binned', 'RFMS_score_binned_WOE', 'ProviderId_WOE',
            'ProviderId_WOE.1', 'ProductId_WOE', 'ProductId_WOE.1',
            'ProductCategory_WOE', 'ProductCategory_WOE.1', 'ChannelId_ChannelId_2',
            'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'ProductId_1',
            'ProductId_2', 'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6',
            'ProductId_7', 'ProductId_8', 'ProductId_9', 'ProductId_10', 'ProductId_11',
            'ProductId_12', 'ProductId_13', 'ProductId_14', 'ProductId_15',
            'ProductId_16', 'ProductId_17', 'ProductId_18', 'ProductId_19',
            'ProductId_20', 'ProductId_21', 'ProductId_22', 'TransactionHour',
            'TransactionDay', 'TransactionMonth'
        ]

        # Add missing columns
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct order
        df = df[expected_features]

        rfms_score = df['RFMS_score'].iloc[0]
        logger.info("Data preprocessed successfully")
        return df, rfms_score
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def predict(model, data: dict) -> dict:
    """Make a prediction using the loaded model and preprocessed data."""
    try:
        # Preprocess the input data
        X, rfms_score = preprocess_data(data)

        # Make prediction
        prediction = model.predict(X)[0]

        # Calculate credit score (0-800 scale)
        credit_score = 800 if prediction == 0 else 400  # Simplified scoring
        logger.info(f"Prediction made: {prediction}, RFMS_score: {rfms_score}, Credit_score: {credit_score}")
        return {
            "prediction": int(prediction),
            "rfms_score": float(rfms_score),
            "credit_score": int(credit_score)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise