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
from sklearn.inspection import PartialDependenceDisplay

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

# Initial placeholder for feature names (will be updated from models)
TRAINING_FEATURE_NAMES = []  # Will be set based on loaded models

def load_models(model_dir='models'):
    global TRAINING_FEATURE_NAMES
    logger.info("Loading saved models")
    try:
        models = {}
        for model_name in ['LogisticRegression', 'RandomForest']:
            model_path = os.path.join(model_dir, f"{model_name}_best_model.pkl")
            logger.info(f"Attempting to load model from: {model_path}")
            models[model_name] = joblib.load(model_path)
            logger.info(f"Loaded {model_name} from {model_path}")
            if model_name == 'RandomForest':
                try:
                    feature_names = getattr(models[model_name], 'feature_names_in_', None)
                    if feature_names is not None:
                        logger.info(f"Random Forest trained feature names: {list(feature_names)} (Count: {len(feature_names)})")
                        TRAINING_FEATURE_NAMES = list(feature_names)  # Update globally
                    else:
                        logger.warning("Feature names not stored in Random Forest model.")
                except Exception as e:
                    logger.warning(f"Could not retrieve feature names from Random Forest: {str(e)}")
        if not TRAINING_FEATURE_NAMES:
            logger.error("No feature names retrieved from models. Using default.")
            TRAINING_FEATURE_NAMES = [
                'ProviderId', 'ProductCategory', 'Amount', 'Value', 'PricingStrategy', 'FraudResult',
                'Total_Transaction_Amount', 'Average_Transaction_Amount', 'Transaction_Count',
                'Std_Transaction_Amount', 'Transaction_Hour', 'Transaction_Day', 'Transaction_Month',
                'Recency', 'RFMS_score', 'RFMS_score_binned', 'RFMS_score_binned_WOE',
                'ProviderId_WOE', 'ProviderId_WOE.1', 'ProductId_WOE', 'ProductId_WOE.1',
                'ProductCategory_WOE', 'ProductCategory_WOE.1', 'ChannelId_ChannelId_2',
                'ChannelId_ChannelId_3', 'ChannelId_ChannelId_5', 'ProductId_1', 'ProductId_2',
                'ProductId_3', 'ProductId_4', 'ProductId_5', 'ProductId_6', 'ProductId_7',
                'ProductId_8', 'ProductId_9', 'ProductId_10', 'ProductId_11', 'ProductId_12',
                'ProductId_13', 'ProductId_14', 'ProductId_15', 'ProductId_16', 'ProductId_17',
                'ProductId_18', 'ProductId_19', 'ProductId_20', 'ProductId_21', 'ProductId_22',
                'TransactionHour', 'TransactionDay', 'TransactionMonth'
            ]  # Fallback (51 features)
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}\n{traceback.format_exc()}")
        return None

def load_or_preprocess_data(data_path, target_column='Label'):
    logger.info("Loading or preprocessing data")
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Raw data columns: {list(data.columns)} (Count: {len(data.columns)})")

        def preprocess_data(data):
            logger.info("Starting data preprocessing")
            try:
                duplicate_columns = data.columns[data.columns.duplicated()]
                if len(duplicate_columns) > 0:
                    logger.warning(f"Found {len(duplicate_columns)} duplicate columns. Removing.")
                    data = data.loc[:, ~data.columns.duplicated()]
                    logger.info(f"After removing duplicates - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                columns_to_drop = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
                existing_columns = [col for col in columns_to_drop if col in data.columns]
                if existing_columns:
                    data = data.drop(columns=existing_columns)
                    logger.info(f"Dropped columns: {existing_columns}")
                    logger.info(f"After dropping columns - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                numeric_cols = data.select_dtypes(include=['number']).columns
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
                categorical_cols = data.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    data[col] = data[col].fillna(data[col].mode()[0])

                categorical_columns = ['ProductId', 'ChannelId']
                existing_categorical = [col for col in categorical_columns if col in data.columns]
                if existing_categorical:
                    data = pd.get_dummies(data, columns=existing_categorical, drop_first=True)
                    logger.info(f"After pd.get_dummies - Columns: {list(data.columns)} (Count: {len(data.columns)})")

                currency_cols = [col for col in data.columns if 'CurrencyCode' in col]
                country_cols = [col for col in data.columns if 'CountryCode' in col]
                cols_to_drop = currency_cols + country_cols
                if cols_to_drop:
                    data = data.drop(columns=cols_to_drop)
                    logger.info(f"Dropped CurrencyCode/CountryCode columns: {cols_to_drop}")
                    logger.info(f"After dropping CurrencyCode/CountryCode - Columns: {list(data.columns)} (Count: {len(data.columns)})")

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
            return None, None, None, None
        logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)} (Count: {len(preprocessed_data.columns)})")

        X = preprocessed_data.drop(columns=[target_column])
        y = preprocessed_data[target_column]
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")  # Validate row count

        if len(X) != len(y):
            logger.error(f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows")
            raise ValueError(f"Row mismatch: X has {len(X)} rows, y has {len(y)} rows")

        logger.info(f"X columns before alignment: {list(X.columns)} (Count: {len(X.columns)})")

        logger.info(f"Training feature names: {TRAINING_FEATURE_NAMES} (Count: {len(TRAINING_FEATURE_NAMES)})")
        missing_cols = set(TRAINING_FEATURE_NAMES) - set(X.columns)
        logger.info(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            X[col] = 0
            logger.info(f"Added missing column: {col}")

        extra_cols = set(X.columns) - set(TRAINING_FEATURE_NAMES)
        logger.info(f"Raw extra columns: {extra_cols}")
        if extra_cols:
            X = X.drop(columns=extra_cols)
            logger.info(f"Dropped extra columns: {extra_cols}")
            logger.info(f"After dropping extra columns - X columns: {list(X.columns)} (Count: {len(X.columns)})")

        try:
            X = X[TRAINING_FEATURE_NAMES]
        except KeyError as e:
            logger.error(f"KeyError during alignment: {str(e)}")
            missing_from_X = [col for col in TRAINING_FEATURE_NAMES if col not in X.columns]
            logger.error(f"Columns missing from X: {missing_from_X}")
            raise
        logger.info(f"After alignment - X columns: {list(X.columns)} (Count: {len(X.columns)})")

        expected_feature_count = len(TRAINING_FEATURE_NAMES)
        if len(X.columns) != expected_feature_count:
            logger.error(f"Alignment failed: Expected {expected_feature_count} features, got {len(X.columns)}")
            raise ValueError(f"Feature count mismatch: Expected {expected_feature_count}, got {len(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info(f"Preprocessed data split: Train {X_train.shape}, Test {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        logger.info(f"X_test columns after split: {list(X_test.columns)} (Count: {len(X_test.columns)})")

        if len(X_test.columns) != expected_feature_count:
            logger.error(f"X_test feature count mismatch: Expected {expected_feature_count}, got {len(X_test.columns)}")
            raise ValueError(f"X_test feature count mismatch: Expected {expected_feature_count}, got {len(X_test.columns)}")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error loading/preprocessing data: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None

def explain_logistic_regression(model, X_train, feature_names, save_dir='explanations'):
    logger.info("Explaining Logistic Regression model")
    try:
        lr_model = model.named_steps['logistic']
        scaler = model.named_steps['scaler']

        # Ensure X_train is aligned with model expectations
        X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_names, index=X_train.index)
        coefficients = lr_model.coef_[0]
        if len(coefficients) != len(feature_names):
            logger.warning(f"Coefficient length ({len(coefficients)}) does not match feature names length ({len(feature_names)}). Padding with zeros.")
            if len(coefficients) > len(feature_names):
                coefficients = coefficients[:len(feature_names)]  # Truncate if more
            else:
                coefficients = np.pad(coefficients, (0, len(feature_names) - len(coefficients)), mode='constant')

        scaled_impact = coefficients * X_train_scaled.std().values
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
    logger.info("Explaining Random Forest model")
    try:
        importances = model.feature_importances_
        if len(importances) != len(feature_names):
            logger.warning(f"Importances length ({len(importances)}) does not match feature names length ({len(feature_names)}). Truncating importances.")
            importances = importances[:len(feature_names)]  # Truncate to match

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

        # SHAP Analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)  # Get SHAP values for all classes
        logger.info(f"SHAP values shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}, X_test shape: {X_test.shape}")

        # Handle SHAP values format
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Older SHAP format: list of arrays (one per class)
            shap_values_class1 = shap_values[1]  # SHAP values for class 1
            logger.info(f"SHAP values for class 1 shape: {shap_values_class1.shape}")
            expected_value = explainer.expected_value[1]  # Base value for class 1
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # Newer SHAP format: single array of shape (n_samples, n_features, n_classes)
            logger.info("Detected newer SHAP output format (numpy array). Extracting SHAP values for class 1.")
            shap_values_class1 = shap_values[:, :, 1]  # Extract SHAP values for class 1
            logger.info(f"SHAP values for class 1 shape: {shap_values_class1.shape}")
            expected_value = explainer.expected_value[1]  # Base value for class 1
        else:
            logger.warning(f"Unexpected SHAP values format: {type(shap_values)}, shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}")
            raise ValueError("Unsupported SHAP output format for binary classification")

        # Validate SHAP values shape
        if shap_values_class1.shape[1] != len(feature_names):
            raise ValueError(f"SHAP values feature count ({shap_values_class1.shape[1]}) does not match feature names count ({len(feature_names)})")
        if shap_values_class1.shape[0] != X_test.shape[0]:
            raise ValueError(f"SHAP values sample count ({shap_values_class1.shape[0]}) does not match X_test sample count ({X_test.shape[0]})")

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values_class1, X_test, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot for Random Forest (Class 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'random_forest_shap_summary.png'))
        plt.show()

        # SHAP Force Plot for a single instance (Fixed for SHAP v0.20+)
        instance_idx = 0
        plt.figure()
        shap.plots.force(
            expected_value,  # Base value as first parameter
            shap_values_class1[instance_idx],  # SHAP values for the instance
            X_test.iloc[instance_idx] if isinstance(X_test, pd.DataFrame) else X_test[instance_idx],  # Features for the instance
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot for Instance {instance_idx} (Class 1)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'random_forest_shap_force_instance_{instance_idx}.png'))
        plt.show()

        # SHAP Dependence Plot for the top feature
        top_feature = importance_df['Feature'].iloc[0]
        plt.figure()
        shap.dependence_plot(
            top_feature,
            shap_values_class1,
            X_test,
            feature_names=feature_names,
            interaction_index='auto',
            show=False
        )
        plt.title(f"SHAP Dependence Plot for {top_feature}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'random_forest_shap_dependence_{top_feature}.png'))
        plt.show()

        # Partial Dependence Plot (PDP) for top 2 features
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
            'shap_values': shap_values_class1
        }
    except Exception as e:
        logger.error(f"Error explaining Random Forest: {str(e)}\n{traceback.format_exc()}")
        return None

def run_explainability_pipeline(data_path, model_dir='models', save_dir='explanations'):
    logger.info("Starting explainability pipeline")
    
    # Load models first to determine feature names
    models = load_models(model_dir)
    if models is None:
        logger.error("Failed to load models. Exiting.")
        return None

    # Proceed with data preprocessing using the correct feature names
    X_train, X_test, y_train, y_test = load_or_preprocess_data(data_path)
    if X_train is None:
        logger.error("Failed to load/preprocess data. Exiting.")
        return None

    feature_names = X_train.columns.tolist()
    explanations = {}
    explanations['LogisticRegression'] = explain_logistic_regression(
        models['LogisticRegression'], X_train, feature_names, save_dir
    )
    explanations['RandomForest'] = explain_random_forest(
        models['RandomForest'], X_train, X_test, feature_names, save_dir
    )

    return explanations

