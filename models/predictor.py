# predictor.py
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from config import logger

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

        # Simulate feature engineering (similar to what was done during training)
        df['Total_Transaction_Amount'] = df['Amount']
        df['Average_Transaction_Amount'] = df['Amount']
        df['Transaction_Count'] = 1
        df['Std_Transaction_Amount'] = 0
        df['Recency'] = (datetime.now() - df['TransactionStartTime']).dt.days
        df['RFMS_score'] = 0.5  # Placeholder

        # Encode categorical variables
        categorical_cols = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
        le = LabelEncoder()
        for col in categorical_cols:
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))

        # One-hot encode ProductId and ChannelId (simulating what was done during training)
        df = pd.get_dummies(df, columns=['ProductId', 'ChannelId'], prefix=['ProductId', 'ChannelId'])

        # Placeholder for WOE encoding (simulating what was done during training)
        for col in ['ProviderId', 'ProductId', 'ChannelId']:
            df[f"{col}_WOE"] = 0.0  # Placeholder for WOE values

        # Drop raw columns that are not used by the model
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
                           'CurrencyCode', 'CountryCode', 'TransactionStartTime']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Define the expected features (51 features as per previous tasks)
        expected_features = [
            'Amount', 'Total_Transaction_Amount', 'Average_Transaction_Amount', 
            'Transaction_Count', 'Std_Transaction_Amount', 'Transaction_Hour', 
            'Transaction_Day', 'Transaction_Month', 'Recency', 'RFMS_score',
            'CurrencyCode_encoded', 'CountryCode_encoded', 'ProductId_encoded', 
            'ChannelId_encoded',
            # One-hot encoded features (example values based on input)
            'ProductId_1', 'ProductId_2', 'ProductId_3',
            'ChannelId_1', 'ChannelId_2', 'ChannelId_3',
            # WOE features
            'ProviderId_WOE', 'ProductId_WOE', 'ChannelId_WOE',
            # Placeholder features to match 51
            *[f'Feature{i}' for i in range(28)]
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