# model_explainability.py (Updated)
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

# Hardcoded feature names from training (temporary for debugging)
TRAINING_FEATURE_NAMES = [
    'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy', 'FraudResult',
    'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count',
    'Std_Transaction_Amount', 'Transaction_Hour', 'Transaction_Day', 'Transaction_Month',
    'Transaction_Year', 'Recency', 'RFMS_score', 'RFMS_score_binned', 'RFMS_score_binned_WOE',
    'ProviderId_WOE', 'ProviderId_WOE.1', 'ProductId_WOE', 'ProductId_WOE.1',
    'ProductCategory_WOE', 'ProductCategory_WOE.1', 'ChannelId_ChannelId_2',
    'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'ProductId_1', 'ProductId_2',
    'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6', 'ProductId_7',
    'ProductId_8', 'ProductId_9', 'ProductId_10', 'ProductId_11', 'ProductId_12',
    'ProductId_13', 'ProductId_14', 'ProductId_15', 'ProductId_16', 'ProductId_17',
    'ProductId_18', 'ProductId_19', 'ProductId_20', 'ProductId_21', 'ProductId_22',
    'TransactionHour', 'TransactionDay', 'TransactionMonth'
]

# Function to load preprocessed data (or re-preprocess if needed)
def load_or_preprocess_data(data_path, target_column='Label'):
    """
    Load preprocessed data or preprocess raw data for explainability, aligning with training features.

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
        logger.info(f"Raw data columns: {list(data.columns)} (Count: {len(data.columns)})")

        # Preprocess data
        def preprocess_data(data):
            logger.info("Starting data preprocessing")
            try:
                # Remove duplicate columns
                duplicate_columns = data.columns[data.columns.duplicated()]
                if len(duplicate_columns) > 0:
                    logger.warning(f"Found {len(duplicate_columns)} duplicate columns. Removing.")
                    data = data.loc[:, ~data.columns.duplicated()]
                    logger.info(f"After removing duplicates - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                # Drop unnecessary columns
                columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
                existing_columns = [col for col in columns_to_drop if col in data.columns]
                if existing_columns:
                    data = data.drop(columns=existing_columns)
                    logger.info(f"Dropped columns: {existing_columns}")
                    logger.info(f"After dropping columns - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                # Handle missing values
                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                categorical_cols = data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    data[col] = data[col].fillna(data[col].mode()[0])

                # Encode categorical variables (ProductId and ChannelId only)
                categorical_columns = ['ProductId', 'ChannelId']
                existing_categorical = [col for col in categorical_columns if col in data.columns]
                if existing_categorical:
                    data = pd.get_dummies(data, columns=existing_categorical, drop_first=True)
                    logger.info(f"After pd.get_dummies - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                # Explicitly drop CurrencyCode and CountryCode columns
                currency_cols = [col for col in data.columns if 'CurrencyCode' in col]
                country_cols = [col for col in data.columns if 'CountryCode' in col]
                cols_to_drop = currency_cols + country_cols
                if cols_to_drop:
                    data = data.drop(columns=cols_to_drop)
                    logger.info(f"Dropped CurrencyCode/CountryCode columns: {cols_to_drop}")
                    logger.info(f"After dropping CurrencyCode/CountryCode - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                # Process TransactionStartTime
                if 'TransactionStartTime' in data.columns:
                    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
                    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
                    data['TransactionDay'] = data['TransactionStartTime'].dt.day
                    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
                    data = data.drop(columns=['TransactionStartTime'])
                    logger.info(f"After processing TransactionStartTime - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                return data
            except Exception as e:
                logger.error(f"Error during preprocessing: {str(e)}\n{traceback.format_exc()}")
                return None

        preprocessed_data = preprocess_data(data)
        if preprocessed_data is None:
            logger.error("Preprocessing failed.")
            return None, None, None, None
        logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)} (Count: {len(preprocessed_data.columns)})")

        # Split the data
        X = preprocessed_data.drop(columns=[target_column])
        y = preprocessed_data[target_column]
        logger.info(f"X columns before alignment: {list(X.columns)} (Count: {len(X.columns)})")

        # Align the features with the training data
        logger.info(f"Training feature names: {TRAINING_FEATURE_NAMES} (Count: {len(TRAINING_FEATURE_NAMES)})")

        # Step 1: Add missing columns from TRAINING_FEATURE_NAMES
        missing_cols = set(TRAINING_FEATURE_NAMES) - set(X.columns)
        logger.info(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            X[col] = 0  # Add missing columns with default value 0
            logger.info(f"Added missing column: {col}")

        # Step 2: Identify extra columns
        extra_cols = set(X.columns) - set(TRAINING_FEATURE_NAMES)
        logger.info(f"Raw extra columns: {extra_cols}")
        # Normalize column names for case-insensitive comparison
        x_cols_normalized = {col.lower().strip() for col in X.columns}
        training_cols_normalized = {col.lower().strip() for col in TRAINING_FEATURE_NAMES}
        extra_cols_normalized = x_cols_normalized - training_cols_normalized
        logger.info(f"Normalized extra columns: {extra_cols_normalized}")
        if extra_cols_normalized:
            extra_cols = [col for col in X.columns if col.lower().strip() in extra_cols_normalized]
            logger.info(f"Dropping extra columns: {extra_cols}")
            X = X.drop(columns=extra_cols, errors='ignore')
            logger.info(f"After dropping extra columns - X columns: {list(X.columns)} (Count: {len(X.columns)})")

        # Step 3: Force alignment to TRAINING_FEATURE_NAMES
        try:
            X = X[TRAINING_FEATURE_NAMES]  # Reorder and subset to exactly 50 features
        except KeyError as e:
            logger.error(f"KeyError during alignment: {str(e)}")
            missing_from_X = [col for col in TRAINING_FEATURE_NAMES if col not in X.columns]
            logger.error(f"Columns missing from X: {missing_from_X}")
            raise
        logger.info(f"After alignment - X columns: {list(X.columns)} (Count: {len(X.columns)})")

        # Step 4: Validate the number of features
        expected_feature_count = len(TRAINING_FEATURE_NAMES)
        if len(X.columns) != expected_feature_count:
            logger.error(f"Alignment failed: Expected {expected_feature_count} features, got {len(X.columns)}")
            raise ValueError(f"Feature count mismatch: Expected {expected_feature_count}, got {len(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Preprocessed data split: Train {X_train.shape}, Test {X_test.shape}")
        logger.info(f"X_test columns after split: {list(X_test.columns)} (Count: {len(X_test.columns)})")

        # Final validation for X_test
        if len(X_test.columns) != expected_feature_count:
            logger.error(f"X_test feature count mismatch: Expected {expected_feature_count}, got {len(X_test.columns)}")
            raise ValueError(f"X_test feature count mismatch: Expected {expected_feature_count}, got {len(X_test.columns)}")

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
            logger.info(f"Attempting to load model from: {model_path}")
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
            # Debug: Log feature names if available (Random Forest)
            if model_name == 'RandomForest':
                try:
                    feature_names = getattr(models[model_name], 'feature_names_in_', None)
                    if feature_names is not None:
                        logger.info(f"Random Forest trained feature names: {list(feature_names)} (Count: {len(feature_names)})")
                    else:
                        logger.warning("Feature names not stored in Random Forest model.")
                except Exception as e:
                    logger.warning(f"Could not retrieve feature names from Random Forest: {str(e)}")
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
        plt.show()

        print("\nLogistic Regression Explanation:")
        print(explanation_df.head(10))
        return explanation_df
    except Exception as e:
        logger.error(f"Error explaining Logistic Regression: {str(e)}\n{traceback.format_exc()}")
        return None

def explain_random_forest(model, X_train, X_test, feature_names, save_dir='explanations'):
    """
    Explain the Random Forest model using feature importance, SHAP values, PDP, and SHAP dependence plots.

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
        plt.show()

        # SHAP Values (Local and Global)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        logger.info(f"SHAP values shape: {np.array(shap_values[1]).shape}, X_test shape: {X_test.shape}")

        plt.figure()
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot for Random Forest (Class 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_forest_shap_summary.png'))
        plt.show()

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
        plt.show()

        # SHAP Dependence Plot for the top feature
        top_feature = importance_df['Feature'].iloc[0]  # Most important feature
        plt.figure()
        shap.dependence_plot(
            top_feature,
            shap_values[1],
            X_test,
            feature_names=feature_names,
            interaction_index='auto',
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {top_feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'random_forest_shap_dependence_{top_feature}.png'))
        plt.show()

        # Partial Dependence Plot for the top 2 features
        from sklearn.inspection import PartialDependenceDisplay
        top_2_features = importance_df['Feature'].head(2).tolist()
        top_2_indices = [feature_names.index(feat) for feat in top_2_features]
        plt.figure()
        PartialDependenceDisplay.from_estimator(
            model,
            X_test,
            features=top_2_indices,
            feature_names=feature_names,
            kind='average'
        )
        plt.title(f"Partial Dependence Plot for {top_2_features}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_forest_pdp.png'))
        plt.show()

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