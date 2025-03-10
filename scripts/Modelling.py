"""
Credit Scoring Model Pipeline (Optimized for Speed, Performance, and Generalization)

This script provides standalone functions to preprocess, split, train, tune, evaluate, and save machine learning 
models for a credit scoring task. It is optimized for speed, high performance, and reduced overfitting risk, 
particularly for Random Forest, with enhanced logging and documentation.
"""

# Import necessary libraries
import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import traceback

# Configure logging with detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up file handlers for info and error logs
base_dir = os.getcwd() if '__file__' not in globals() else os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

info_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
info_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Define standalone functions

def preprocess_data(data):
    """
    Preprocess the dataset efficiently by handling duplicates, missing values, categorical encoding, and datetime features.

    Parameters:
    - data (pd.DataFrame): Raw input dataset.

    Returns:
    - pd.DataFrame: Preprocessed dataset, or None if an error occurs.

    Notes:
    - Logs detailed steps and dataset stats for monitoring.
    - Uses median for numeric and mode for categorical missing values to maintain robustness.
    """
    logger.info("Starting data preprocessing")
    try:
        logger.info(f"Initial dataset shape: {data.shape}")
        
        # Remove duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()]
        if len(duplicate_columns) > 0:
            logger.warning(f"Found {len(duplicate_columns)} duplicate columns: {list(duplicate_columns)}. Removing.")
            data = data.loc[:, ~data.columns.duplicated()]

        # Drop unnecessary columns
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        existing_columns = [col for col in columns_to_drop if col in data.columns]
        if existing_columns:
            data = data.drop(columns=existing_columns)
            logger.info(f"Dropped columns: {existing_columns}")

        # Handle missing values
        numeric_cols = data.select_dtypes(include=['number']).columns
        missing_numeric = data[numeric_cols].isnull().sum().sum()
        if missing_numeric > 0:
            logger.info(f"Filling {missing_numeric} missing numeric values with median")
            data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        categorical_cols = data.select_dtypes(include=['object']).columns
        missing_categorical = data[categorical_cols].isnull().sum().sum()
        if missing_categorical > 0:
            logger.info(f"Filling {missing_categorical} missing categorical values with mode")
            for col in categorical_cols:
                data[col] = data[col].fillna(data[col].mode()[0])

        # Encode categorical variables
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
        existing_categorical = [col for col in categorical_columns if col in data.columns]
        if existing_categorical:
            logger.info(f"Encoding categorical columns: {existing_categorical}")
            data = pd.get_dummies(data, columns=existing_categorical, drop_first=True)

        # Extract datetime features
        if 'TransactionStartTime' in data.columns:
            logger.info("Extracting datetime features from 'TransactionStartTime'")
            data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
            data['TransactionHour'] = data['TransactionStartTime'].dt.hour
            data['TransactionDay'] = data['TransactionStartTime'].dt.day
            data['TransactionMonth'] = data['TransactionStartTime'].dt.month
            data = data.drop(columns=['TransactionStartTime'])

        logger.info(f"Preprocessed dataset shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}\n{traceback.format_exc()}")
        return None

def split_the_data(data, target_column='Label', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets with stratification.

    Parameters:
    - data (pd.DataFrame): Preprocessed dataset.
    - target_column (str): Name of the target variable (default: 'Label').
    - test_size (float): Proportion of data for testing (default: 0.2).
    - random_state (int): Seed for reproducibility (default: 42).

    Returns:
    - tuple: (X_train, X_test, y_train, y_test), or (None, None, None, None) if an error occurs.
    """
    logger.info("Starting data splitting")
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        logger.info(f"Dataset size: {data.shape}, Target distribution: {y.value_counts(normalize=True).to_dict()}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None

def define_models():
    """
    Define machine learning models with regularization to prevent overfitting.

    Returns:
    - dict: Dictionary of model names mapped to their instances or pipelines.

    Notes:
    - RandomForest includes max_features='sqrt' to limit feature usage per split, reducing overfitting.
    """
    logger.info("Defining models")
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(solver='lbfgs', max_iter=500, random_state=42))
        ]),
        'RandomForest': RandomForestClassifier(
            random_state=42, n_jobs=-1, max_features='sqrt'  # Regularization to limit features per split
        )
    }
    logger.info("Models defined: LogisticRegression (with scaling), RandomForest (with regularization)")
    return models

def define_hyperparameter_grids():
    """
    Define hyperparameter grids for tuning, with constraints to reduce Random Forest overfitting.

    Returns:
    - dict: Dictionary of model names mapped to their hyperparameter grids.

    Notes:
    - RandomForest uses shallower max_depth and min_samples_leaf to prevent overfitting.
    """
    logger.info("Defining hyperparameter grids")
    param_grids = {
        'LogisticRegression': {
            'logistic__C': [0.1, 1, 10],  # Regularization strength
            'logistic__penalty': ['l2']   # lbfgs supports l2 only
        },
        'RandomForest': {
            'n_estimators': [50, 100],    # Number of trees
            'max_depth': [5, 10, 15],     # Shallower trees to reduce overfitting
            'min_samples_split': [5],     # Minimum samples to split
            'min_samples_leaf': [2, 4]    # Minimum samples per leaf for regularization
        }
    }
    logger.info("Hyperparameter grids defined with regularization for RandomForest")
    return param_grids

def perform_grid_search(models, param_grids, X_train, y_train):
    """
    Perform randomized search to tune hyperparameters efficiently while avoiding overfitting.

    Parameters:
    - models (dict): Dictionary of model names and their instances.
    - param_grids (dict): Dictionary of model names and their hyperparameter grids.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.

    Returns:
    - dict: Dictionary of model names mapped to their best estimators, or None if an error occurs.
    """
    logger.info("Starting Randomized Search for hyperparameter tuning")
    best_models = {}
    try:
        for name, model in models.items():
            logger.info(f"Performing Randomized Search for {name}")
            search = RandomizedSearchCV(
                model,
                param_grids[name],
                n_iter=5,  # Reduced iterations for speed, sufficient for small grid
                cv=3,      # 3 folds for speed; increase to 5 if overfitting persists
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=1  # Adds progress output
            )
            search.fit(X_train, y_train)
            best_models[name] = search.best_estimator_
            logger.info(f"{name} best parameters: {search.best_params_}")
            logger.info(f"{name} best CV ROC-AUC: {search.best_score_:.4f}")
        logger.info("Randomized Search completed successfully")
        return best_models
    except Exception as e:
        logger.error(f"Error during Randomized Search: {str(e)}\n{traceback.format_exc()}")
        return None

def evaluate_best_models(best_models, X_test, y_test, X_train, y_train):
    """
    Evaluate the best models on test data and compare with cross-validation to detect overfitting.

    Parameters:
    - best_models (dict): Dictionary of model names and their best estimators.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing labels.
    - X_train (pd.DataFrame): Training features (for CV).
    - y_train (pd.Series): Training labels (for CV).

    Returns:
    - dict: Dictionary of model names mapped to their evaluation metrics, or None if an error occurs.

    Notes:
    - Adds 5-fold CV on training data to check generalization and logs results.
    """
    logger.info("Starting evaluation of best models")
    try:
        results = {}
        for name, model in best_models.items():
            logger.info(f"Evaluating {name}")
            # Test set evaluation
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_prob)
            }
            results[name] = metrics
            print(f"\nMetrics for {name} (Test Set):")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                logger.info(f"{name} Test {metric}: {value:.4f}")

            # Cross-validation on training set to check overfitting
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            logger.info(f"{name} 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"{name} 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        logger.info("Model evaluation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}\n{traceback.format_exc()}")
        return None

def save_best_models(best_models, save_dir='models'):
    """
    Save the best models to disk for later use.

    Parameters:
    - best_models (dict): Dictionary of model names and their best estimators.
    - save_dir (str): Directory to save the models (default: 'models').

    Returns:
    - None
    """
    logger.info("Starting process to save best models")
    try:
        os.makedirs(save_dir, exist_ok=True)
        for name, model in best_models.items():
            file_path = os.path.join(save_dir, f"{name}_best_model.pkl")
            joblib.dump(model, file_path)
            logger.info(f"Saved {name} to {file_path}")
        logger.info("All best models saved successfully")
    except Exception as e:
        logger.error(f"Failed to save models: {str(e)}\n{traceback.format_exc()}")

# 