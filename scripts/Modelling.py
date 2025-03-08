"""
Credit Scoring Model Pipeline

This script provides standalone functions to preprocess, split, train, tune, evaluate, and save machine learning 
models for a credit scoring task. It is designed to be used in a Jupyter notebook, allowing each step to be executed 
and inspected individually in separate cells.
"""

# Import necessary libraries
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Determine the base directory (handles both script and notebook environments)
if '__file__' in globals():
    base_dir = os.path.dirname(os.path.abspath(__file__))
else:
    base_dir = os.getcwd()
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Set up file handlers for info and error logs
info_handler = logging.FileHandler(os.path.join(log_dir, 'info.log'))
info_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler(os.path.join(log_dir, 'error.log'))
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Define standalone functions

def preprocess_data(data):
    """
    Preprocess the dataset by checking for duplicate columns, handling missing values, encoding categorical variables,
    and extracting datetime features.

    Parameters:
    - data (pd.DataFrame): Raw dataset to preprocess.

    Returns:
    - pd.DataFrame: Preprocessed dataset, or None if an error occurs.
    """
    logger.info("Starting data preprocessing")
    try:
        # Check for duplicate columns
        duplicate_columns = data.columns[data.columns.duplicated()]
        if not duplicate_columns.empty:
            logger.warning(f"Duplicate columns found: {duplicate_columns}. Removing duplicates.")
            data = data.loc[:, ~data.columns.duplicated()]
        
        # Drop unnecessary columns if they exist
        columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        logger.info("Dropped unnecessary columns")

        # Handle missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            logger.info(f"Missing values found in columns: {missing_values[missing_values > 0].index.tolist()}")
            # Fill missing values with median for numerical columns
            for col in data.select_dtypes(include=['number']).columns:
                data[col] = data[col].fillna(data[col].median())
            # Fill missing values with mode for categorical columns
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].fillna(data[col].mode()[0])
            logger.info("Handled missing values")

        # Encode categorical variables
        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
        data = pd.get_dummies(data, columns=[col for col in categorical_columns if col in data.columns], drop_first=True)
        logger.info("Encoded categorical variables")

        # Extract datetime features from TransactionStartTime
        if 'TransactionStartTime' in data.columns:
            logger.info("Extracting datetime features from TransactionStartTime")
            data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
            data['TransactionHour'] = data['TransactionStartTime'].dt.hour
            data['TransactionDay'] = data['TransactionStartTime'].dt.day
            data['TransactionMonth'] = data['TransactionStartTime'].dt.month
            data['TransactionWeekday'] = data['TransactionStartTime'].dt.weekday
            data = data.drop(columns=['TransactionStartTime'])
            logger.info("Datetime features extracted and original column dropped")

        logger.info("Data preprocessing completed successfully")
        return data
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None

def split_the_data(data, target_column='Label', test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - data (pd.DataFrame): Preprocessed dataset.
    - target_column (str): Name of the target variable column (default: 'Label').
    - test_size (float): Proportion of data for the test set (default: 0.2).
    - random_state (int): Seed for reproducibility (default: 42).

    Returns:
    - tuple: (X_train, X_test, y_train, y_test), or None if an error occurs.
    """
    logger.info("Starting data splitting")
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info("Data split into training and testing sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        return None, None, None, None

def define_models():
    """
    Define the machine learning models to be used, with Logistic Regression in a pipeline for scaling.

    Returns:
    - dict: Dictionary of model names mapped to their instances or pipelines.
    """
    logger.info("Defining machine learning models")
    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('logistic', LogisticRegression(solver='liblinear', max_iter=1000, random_state=42))
        ]),
        'RandomForest': RandomForestClassifier(random_state=42)
    }
    logger.info("Models defined: LogisticRegression (with scaling), RandomForest")
    return models

def define_hyperparameter_grids():
    """
    Define hyperparameter grids for model tuning.

    Returns:
    - dict: Dictionary of model names mapped to their hyperparameter grids.
    """
    logger.info("Defining hyperparameter grids")
    param_grids = {
        'LogisticRegression': {
            'logistic__C': [0.1, 1, 10],
            'logistic__penalty': ['l1', 'l2']
        },
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
    logger.info("Hyperparameter grids defined for LogisticRegression and RandomForest")
    return param_grids

def perform_grid_search(models, param_grids, X_train, y_train):
    """
    Perform grid search to find the best hyperparameters for each model.

    Parameters:
    - models (dict): Dictionary of model names and their instances.
    - param_grids (dict): Dictionary of model names and their hyperparameter grids.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training labels.

    Returns:
    - dict: Dictionary of model names mapped to their best estimators, or None if an error occurs.
    """
    logger.info("Starting Grid Search for hyperparameter tuning")
    best_models = {}
    try:
        for name, model in models.items():
            logger.info(f"Performing Grid Search for {name}")
            grid_search = GridSearchCV(
                model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_models[name] = grid_search.best_estimator_
            logger.info(f"{name} best parameters: {grid_search.best_params_}")
        logger.info("Grid Search completed successfully")
        return best_models
    except Exception as e:
        logger.error(f"Error during Grid Search: {e}")
        return None

def evaluate_best_models(best_models, X_test, y_test):
    """
    Evaluate the best models on the test data and display metrics.

    Parameters:
    - best_models (dict): Dictionary of model names and their best estimators.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing labels.

    Returns:
    - dict: Dictionary of model names mapped to their evaluation metrics, or None if an error occurs.
    """
    logger.info("Starting evaluation of best models on test data")
    try:
        results = {}
        for name, model in best_models.items():
            logger.info(f"Evaluating {name}")
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
            print(f"\nMetrics for {name}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            logger.info(f"Evaluation completed for {name}")
        logger.info("Model evaluation completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"Created save directory: {save_dir}")
        for name, model in best_models.items():
            file_path = os.path.join(save_dir, f"{name}_best_model.pkl")
            joblib.dump(model, file_path)
            logger.info(f"Saved {name} to {file_path}")
        logger.info("All best models saved successfully")
    except Exception as e:
        logger.error(f"Failed to save models: {e}")

