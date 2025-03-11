import sys
import os
import joblib
import pandas as pd
import numpy as np
import traceback

# Adjust sys.path to include the root directory (Render uses flat structure)
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from credit_scoring_app.config import logger

def load_model(model_path: str):
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise

def preprocess_data(data: dict, target_label: str = "TRUE") -> tuple[pd.DataFrame, float]:
    try:
        logger.info(f"Preprocessing data: {data}")
        df = pd.DataFrame([data])

        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
        df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
        df['Transaction_Day'] = df['TransactionStartTime'].dt.day
        df['Transaction_Month'] = df['TransactionStartTime'].dt.month
        df['Transaction_Year'] = df['TransactionStartTime'].dt.year
        df['TransactionHour'] = df['Transaction_Hour']
        df['TransactionDay'] = df['Transaction_Day']
        df['TransactionMonth'] = df['Transaction_Month']

        current_date = pd.Timestamp.now(tz='UTC')
        df['Recency'] = (current_date - df['TransactionStartTime']).dt.days

        df['Total_Transaction_Amount'] = df['Amount']
        df['Average_Transaction_Amount'] = df['Amount']
        df['Transaction_Count'] = 1
        df['Std_Transaction_Amount'] = 0

        df['RFMS_score'] = (1 / (df['Recency'] + 1) * 0.4) + (df['Transaction_Count'] * 0.3) + (df['Total_Transaction_Amount'] * 0.3)
        df['RFMS_score'] = df['RFMS_score'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['RFMS_score_binned'] = 0

        df['ProviderId'] = 0
        df['ProductCategory'] = 0
        df['Value'] = 0
        df['PricingStrategy'] = 0

        # Ensure FraudResult is included
        if 'FraudResult' not in df.columns:
            logger.warning("FraudResult not provided in input data. Defaulting to 0.")
            df['FraudResult'] = 0

        # Set WOE features based on target_label
        if target_label == "TRUE":
            df['RFMS_score_binned_WOE'] = -0.071
            df['ProviderId_WOE'] = -0.41361
            df['ProviderId_WOE.1'] = -0.41361
            df['ProductId_WOE'] = 0.214869
            df['ProductId_WOE.1'] = 0.214869
            df['ProductCategory_WOE'] = -0.5793
            df['ProductCategory_WOE.1'] = 0.109343
        else:  # FALSE
            df['RFMS_score_binned_WOE'] = -0.071
            df['ProviderId_WOE'] = -0.41361
            df['ProviderId_WOE.1'] = -0.41361
            df['ProductId_WOE'] = 0.3044295
            df['ProductId_WOE.1'] = 0.3044295
            df['ProductCategory_WOE'] = -0.5793
            df['ProductCategory_WOE.1'] = 0.109343

        df = pd.get_dummies(df, columns=['ChannelId'], prefix='ChannelId_ChannelId')
        for value in [2, 3, 5]:
            col_name = f'ChannelId_ChannelId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        df = pd.get_dummies(df, columns=['ProductId'], prefix='ProductId')
        for value in range(1, 23):
            col_name = f'ProductId_{value}'
            if col_name not in df.columns:
                df[col_name] = 0

        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                          'CurrencyCode', 'CountryCode', 'TransactionStartTime']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Use the exact feature names and order from the model
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

        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_features]
        rfms_score = df['RFMS_score'].iloc[0]
        logger.info("Data preprocessed successfully")
        return df, rfms_score
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}")
        raise

def predict(model, data: dict, target_label: str = "TRUE") -> dict:
    try:
        X, rfms_score = preprocess_data(data, target_label)
        logger.info(f"Preprocessed data shape: {X.shape}")

        prediction = model.predict(X)[0]
        logger.info(f"Raw prediction: {prediction}")

        return {
            "prediction": int(prediction),
            "rfms_score": float(rfms_score)
        }
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}\n{traceback.format_exc()}")
        raise