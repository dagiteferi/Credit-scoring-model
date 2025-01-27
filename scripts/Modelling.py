import logging, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

logger.addHandler(info_handler)
logger.addHandler(error_handler)


def load_data(path):
    logger.info("importing the data")
    try:
        logger.info("loading the data")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"error occurred while loading the data {e}")


def preprocess_data(data):
    logger.info("Preprocessing the data")
    try:
        data = data.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId'])
        data['Std_Transaction_Amount'].fillna(data['Std_Transaction_Amount'].median(), inplace=True)

        categorical_columns = ['CurrencyCode', 'CountryCode', 'ProductId', 'ChannelId']
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

        if 'TransactionStartTime' in data.columns:
            logger.info("Extracting datetime features from 'TransactionStartTime'")
            data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
            data['TransactionHour'] = data['TransactionStartTime'].dt.hour
            data['TransactionDay'] = data['TransactionStartTime'].dt.day
            data['TransactionMonth'] = data['TransactionStartTime'].dt.month
            data['TransactionWeekday'] = data['TransactionStartTime'].dt.weekday
            data = data.drop(columns=['TransactionStartTime'])

        logger.info("Data preprocessing completed")
        return data
    except Exception as e:
        logger.error(f"Error occurred while preprocessing the data: {e}")
        return None


def split_the_data(data):
    logger.info("splitting the data")
    try:
        X = data.drop(columns=['Label'])
        y = data['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.info(f"error occurred {e}")


def tain_the_models(X_train, y_train, X_test):
    logger.info("train the model")
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        logger.info("Initializing the model")
        logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        random_forest_model = RandomForestClassifier(random_state=42)
        logger.info("training the model with our data")
        logistic_model.fit(X_train_scaled, y_train)
        random_forest_model.fit(X_train, y_train)
        return logistic_model, random_forest_model
    except Exception as e:
        logger.error(f"error occurred {e}")


def define_hyperparameter_grids():
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    return param_grids


def perform_grid_search(models, param_grids, X_train, y_train):
    logging.info("Performing Grid Search for each model.")
    best_models = {}
    for name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        logging.info(f"{name} Best Parameters: {grid_search.best_params_}")
    return best_models


def evaluate_models(model, X_test, y_test):
    logger.info("Evaluate the models")
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")

        return y_pred
    except Exception as e:
        logger.error(f"error occurs {e}")
