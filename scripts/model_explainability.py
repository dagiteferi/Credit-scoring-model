# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import logging
import os
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

base_dir = os.getcwd()
log_dir = os.path.join(base_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

info_handler = logging.FileHandler(os.path.join(log_dir, 'explainability_info.log'))
info_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler(os.path.join(log_dir, 'explainability_error.log'))
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)

# Function to load preprocessed data (or re-preprocess if needed)
def load_or_preprocess_data(data_path, target_column='Label'):
    """
    Load preprocessed data or preprocess raw data for explainability.

    Parameters:
    - data_path (str): Path to the raw or preprocessed data CSV.
    - target_column (str): Name of the target column.

    Returns:
    - X_train, X_test, y_train, y_test: Preprocessed train/test splits.
    """
    logger.info("Loading or preprocessing data")
    try:
        # Load data
        data = pd.read_csv(data_path)
        
        # Preprocess (reusing your original preprocess_data function)
        def preprocess_data(data):
            logger.info("Starting data preprocessing")
            try:
                duplicate_columns = data.columns[data.columns.duplicated()]
                if len(duplicate_columns) > 0:
                    logger.warning(f"Found {len(duplicate_columns)} duplicate columns. Removing.")
                    data = data.loc[:, ~data.columns.duplicated()]

                columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
                existing_columns = [col for col in columns_to_drop if col in data.columns]
                if existing_columns:
                    data = data.drop(columns=existing_columns)
                    logger.info(f"Dropped columns: {existing_columns}")

                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                categorical_cols = data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    data[col] = data[col].fillna(data[col].mode()[0])

                categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
                existing_categorical = [col for col in categorical_columns if col in data.columns]
                if existing_categorical:
                    data = pd.get_dummies(data, columns=existing_categorical, drop_first=True)

                if 'TransactionStartTime' in data.columns:
                    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
                    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
                    data['TransactionDay'] = data['TransactionStartTime'].dt.day
                    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
                    data = data.drop(columns=['TransactionStartTime'])

                return data
            except Exception as e:
                logger.error(f"Error during preprocessing: {str(e)}\n{traceback.format_exc()}")
                return None

        preprocessed_data = preprocess_data(data)
        if preprocessed_data is None:
            return None, None, None, None

        # Split the data
        X = preprocessed_data.drop(columns=[target_column])
        y = preprocessed_data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Preprocessed data split: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None

# Load saved models
def load_models(model_dir='models'):
    """
    Load saved models for explainability.

    Parameters:
    - model_dir (str): Directory where models are saved.

    Returns:
    - dict: Dictionary of model names and their loaded instances.
    """
    logger.info("Loading saved models")
    try:
        models = {}
        for model_name in ['LogisticRegression', 'RandomForest']:
            model_path = os.path.join(model_dir, f"{model_name}_best_model.pkl")
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
        return None

# Explainability Functions
def explain_logistic_regression(model, X_train, feature_names, save_dir='explanations'):
    """
    Explain the Logistic Regression model by analyzing its coefficients.

    Parameters:
    - model: Trained LogisticRegression model (within a Pipeline).
    - X_train (pd.DataFrame): Training features (scaled, if applicable).
    - feature_names (list): Names of the features.
    - save_dir (str): Directory to save the explanation plot.

    Returns:
    - pd.DataFrame: DataFrame with feature names and their coefficients.
    """
    logger.info("Explaining Logistic Regression model")
    try:
        lr_model = model.named_steps['logistic']
        scaler = model.named_steps['scaler']

        coefficients = lr_model.coef_[0]
        scaled_impact = coefficients * X_train.std().values
        explanation_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Scaled_Impact': scaled_impact
        })
        explanation_df = explanation_df.sort_values(by='Scaled_Impact', key=abs, ascending=False)

        logger.info("Top 5 impactful features for Logistic Regression:")
        logger.info(explanation_df.head().to_string())

        plt.figure(figsize=(10, 6))
        plt.barh(explanation_df['Feature'].head(10), explanation_df['Scaled_Impact'].head(10))
        plt.xlabel('Scaled Impact on Prediction')
        plt.title('Logistic Regression Feature Impact')
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'logistic_regression_explanation.png'))
        plt.close()

        print("\nLogistic Regression Explanation:")
        print(explanation_df.head(10))
        return explanation_df
    except Exception as e:
        logger.error(f"Error explaining Logistic Regression: {str(e)}\n{traceback.format_exc()}")
        return None

def explain_random_forest(model, X_train, X_test, feature_names, save_dir='explanations'):
    """
    Explain the Random Forest model using feature importance and SHAP values.

    Parameters:
    - model: Trained RandomForestClassifier model.
    - X_train (pd.DataFrame): Training features (for SHAP explainer).
    - X_test (pd.DataFrame): Test features (for explanations).
    - feature_names (list): Names of the features.
    - save_dir (str): Directory to save explanation plots.

    Returns:
    - dict: Feature importance DataFrame and SHAP values.
    """
    logger.info("Explaining Random Forest model")
    try:
        # Feature Importance (Global)
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        logger.info("Top 5 important features for Random Forest:")
        logger.info(importance_df.head().to_string())

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'].head(10), importance_df['Importance'].head(10))
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'random_forest_importance.png'))
        plt.close()

        # SHAP Values (Local and Global)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure()
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot for Random Forest (Class 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_forest_shap_summary.png'))
        plt.close()

        instance_idx = 0
        plt.figure()
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1][instance_idx],
            X_test.iloc[instance_idx],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot for Instance {instance_idx} (Class 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'random_forest_shap_force_instance_{instance_idx}.png'))
        plt.close()

        print("\nRandom Forest Feature Importance:")
        print(importance_df.head(10))

        return {
            'feature_importance': importance_df,
            'shap_values': shap_values
        }
    except Exception as e:
        logger.error(f"Error explaining Random Forest: {str(e)}\n{traceback.format_exc()}")
        return None

# Main Explainability Pipeline
def run_explainability_pipeline(data_path, model_dir='models', save_dir='explanations'):
    """
    Run the explainability pipeline on saved models.

    Parameters:
    - data_path (str): Path to the raw or preprocessed data CSV.
    - model_dir (str): Directory where models are saved.
    - save_dir (str): Directory to save explanation plots.

    Returns:
    - dict: Explanations for each model.
    """
    logger.info("Starting explainability pipeline")
    
    # Load or preprocess data
    X_train, X_test, y_train, y_test = load_or_preprocess_data(data_path)
    if X_train is None:
        logger.error("Failed to load/preprocess data. Exiting.")
        return None

    # Load models
    models = load_models(model_dir)
    if models is None:
        logger.error("Failed to load models. Exiting.")
        return None

    # Explain models
    feature_names = X_train.columns.tolist()
    explanations = {}
    explanations['LogisticRegression'] = explain_logistic_regression(
        models['LogisticRegression'], X_train, feature_names, save_dir
    )
    explanations['RandomForest'] = explain_random_forest(
        models['RandomForest'], X_train, X_test, feature_names, save_dir
    )

    return explanations

